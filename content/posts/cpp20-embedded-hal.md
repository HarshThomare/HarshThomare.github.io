---
title: "The HAL You Can Compile Twice"
date: 2025-04-09
draft: false
categories: ["Embedded", "C++"]
tags: ["C++20", "concepts", "STM32", "testing"]
summary: "A concept-checked UART driver can be compiled once against a host mock and once against memory-mapped registers, and the tradeoff against a virtual HAL is code size versus a call the compiler can see."
description: "C++20 concepts as the constraint on an embedded HAL, with a mock that stays out of the firmware image."
---

A UART driver that has to compile for a host test and for flash needs two different bodies for `write()` and `read()`. In the test, the body stores bytes in a host container. On the device, it loads a status register. The source should be the same in both builds, and the hot path should not go through a vtable. Some code polls often, and the flash budget is on the order of a few hundred kilobytes, where a second copy of a large driver is not free. Geoffrey Hunter's [notes on C++ in embedded systems](https://blog.mbedded.ninja/programming/languages/c-plus-plus/cpp-on-embedded-systems/) already cover the part of C++ that is safe on bare metal, so I will stay with the driver.

## The C habit already contains the object

A C UART driver takes a struct of registers as its first argument. That is already an object, and most of us write it without thinking of it that way. The usual answers to the two-body problem are a function pointer stored in the struct, so the test can point it at a fake, and a pile of `#ifdef`s that swaps the bodies at build time. Both work. Both also hide something from the compiler. On the device, the address of the status register is a constant. Neither answer lets the compiler treat it as one at the call site.

## A concept is only a predicate

The C++20 version starts with a concept. It generates no code. It is a compile-time statement about what a type must be able to do.

```cpp
template <typename T>
concept UartHal = requires(T& uart, std::uint8_t byte) {
    { uart.write(byte) } -> std::same_as<bool>;
    { uart.bytes_available() } -> std::same_as<std::size_t>;
    { uart.read() } -> std::same_as<std::uint8_t>;
};

template <UartHal Hal>
class LineReader {
 public:
    explicit LineReader(Hal& hal) : hal_(hal) {}

    template <std::size_t N>
    bool poll_line(std::array<char, N>& out) {
        std::size_t n = 0;
        while (hal_.bytes_available() > 0 && n + 1 < N) {
            char c = static_cast<char>(hal_.read());
            if (c == '\n') {
                out[n] = '\0';
                return true;
            }
            out[n++] = c;
        }
        return false;
    }

 private:
    Hal& hal_;
};
```

The requires-expression is checked when `LineReader` is instantiated. If I hand it a mock that forgot `read()`, the error appears at the interface. It names the concept and the missing expression, and it does not appear inside `poll_line`. That is the practical gain over the CRTP approach we had before C++20. The old error named some internal member three templates deep. You had to work backwards to find which type was at fault.

Concepts did not invent static polymorphism. CRTP and plain templates had it for years. What concepts did was make the constraint readable. I think a constraint that a new person can read is most of what makes an interface worth having.

## Only the register type reaches the image

The snippets assume `<cstdint>`, `<array>`, `<string>`, `<vector>`, and `<concepts>`.

```cpp
// Host test only. std::string does not belong in the flash image.
struct MockUart {
    std::string rx;
    std::vector<std::uint8_t> tx;
    std::size_t cursor = 0;

    bool write(std::uint8_t byte) {
        tx.push_back(byte);
        return true;
    }

    std::size_t bytes_available() const { return rx.size() - cursor; }

    std::uint8_t read() { return static_cast<std::uint8_t>(rx[cursor++]); }
};

// The address of SR is a constant at the call site if the compiler inlines this.
struct Stm32Uart {
    volatile std::uint32_t* sr;
    volatile std::uint32_t* dr;
    static constexpr std::uint32_t rxne = 1u << 5;
    static constexpr std::uint32_t txe = 1u << 7;

    bool write(std::uint8_t byte) {
        if ((*sr & txe) == 0) {
            return false;
        }
        *dr = byte;
        return true;
    }

    std::size_t bytes_available() const { return (*sr & rxne) != 0 ? 1u : 0u; }

    std::uint8_t read() { return static_cast<std::uint8_t>(*dr); }
};
```

`MockUart` keeps its bytes in host containers, a `std::string` of pending input and a `std::vector` of everything written. Those containers must never be linked into the firmware. They are not, because the firmware never instantiates `LineReader<MockUart>`.

`Stm32Uart` holds pointers to the status and data registers. Because the driver is a template, a call to `write()` in the firmware instantiation is visible as ordinary code. The compiler can inline it. Once it is inlined, a constant register address can be folded into the load. A virtual call fetches a function pointer through the vptr and calls through it. In general the compiler cannot see past that to the register.

`bytes_available()` returns 0 or 1 because it reads a status bit, and a status bit is not a FIFO depth. A driver for a part with a deeper receive buffer would answer differently. I would not ship it as it stands.

## A concept gives two drivers, a virtual HAL gives one

<figure>
  <img src="/images/hal-concepts.svg" alt="Left: one LineReader calls through a UART vtable to either a mock or a register driver. Right: a concept checks two monomorphized LineReaders, and the device copy sees a constant status-register address.">
  <figcaption>A virtual HAL is one type, chosen at runtime. A concept is a predicate, and each model of it is compiled separately. The mock's containers stay in the host binary.</figcaption>
</figure>

The left side of the figure is the virtual design. There is one `LineReader`, one type, and an indirect call to whichever UART it was given. That is what you want when a protocol object must point at one of several UARTs chosen at runtime. The right side has two monomorphized drivers, each compiled for exactly one HAL. The mock lives in the host test, and the firmware image never contains `std::vector`.

## The tradeoff is code size against a call the compiler can see

The answer depends on how often the code runs and how big it is.

Virtual dispatch costs an indirect call and a vptr, and it blocks constant propagation of the register address. On a housekeeping task that runs once a second, the indirect call and the lost fold do not matter. A virtual interface is then the simpler program. On a tight poll, the lost fold matters more than the vptr. The vptr is a few bytes. The lost fold is what changes what the compiler is able to do with the loop.

Templates cost code size. With the design above there are two HALs and therefore two copies of `LineReader`. If `LineReader` is small, as this one is, the copy is cheaper than the indirect call. The host copy is not in the image anyway. The harder case is when someone templates an entire application object on the HAL. The binary then grows by a second copy of the application. On a 256 KB part that is a bad trade even if every individual call reads more cleanly. For that reason I think the concept belongs on the small driver and not on the whole firmware. The place to stop is the point where the thing being duplicated is no longer small.

## The concept does not supply time

A mock that only records `write()` is a weak mock for control code. Concepts did not solve testing. A UART that feeds a parser can be tested with a string. That is why `MockUart` is enough for `LineReader`. A control loop that reads a sensor and writes a duty cycle is different. It needs a scripted timeline: at this step the register reads this value, at the next step it reads that one, and the test checks the duty that came out.

You can put that timeline inside a type that models the same concept. The timeline is still something you write. The concept constrains the shape of the interface and says nothing about when a read returns what. If a test needs time, the mock has to carry it.

## A table of devices still wants a vtable

Suppose one protocol object must bind to USART1 or USART2 depending on a board revision that is read at boot. A template cannot express that without also containing both instantiations and a branch that selects between them. That branch is fine. In that situation a virtual interface is the shorter and clearer program. Pretending that concepts replace runtime polymorphism is how this style gets a bad name.

Hunter's caution about not going wild with polymorphism is a warning against reaching for virtual functions by reflex. It is equally a warning against reaching for templates by reflex.

## The firmware pays for one extra instantiation

The concept is the contract that lets the driver compile twice. In one build the body is a mock backed by host containers. In the other the body is a volatile load. The firmware pays for the second instantiation only, and only for the driver that was actually templated.
