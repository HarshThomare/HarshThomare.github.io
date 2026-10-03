---
title: "Reflection Belongs on the Host, Next to the Datasheet"
date: 2026-09-21
draft: false
categories: ["Embedded", "C++"]
tags: ["C++26", "reflection", "registers", "testing"]
summary: "The embedded use of C++26 reflection I actually want is a host-side check that a register struct and its legal-bit table still name the same fields, so the mock and the datasheet cannot drift."
description: "A narrow use of C++26 reflection: generate nothing the device compiler cannot already build, and fail the host build when a register map drifts."
---

## The two descriptions of the same registers drift.

A register map is a struct of `volatile uint32_t`. A test mock is a second description of the same registers, usually a switch on an offset or a set of bit masks in a test file. Someone adds `CCR2` to the struct, or widens a field, and the mock still accepts a write that the reference manual reserves. The tests stay green because none of them mention the new field. The silicon then does something dull and bad, and nothing in the build noticed that the two descriptions had stopped agreeing.

The use of C++26 reflection I want is a host-side check that the struct and a table of legal bits still name the same fields, so that the mock and the datasheet cannot drift apart without the build going red. The device compiler keeps building ordinary C++20 and never sees any of it.

The mechanism is small. In the wording of P2996, the prefix operator `^^` turns a type or a member into a `consteval` value of type `std::meta::info`, and a splice, written `[: refl :]`, turns such a value back into code. Everything else is a function that takes one of those values and answers a question about it, and I will name those functions where the walk needs them. The usual pitch for all this is serialization: walk any struct and print it. That is a real use, but it is not the one that fits a microcontroller. Firmware compilers lag, and I do not want the idea to wait for the STM32 toolchain. By "the host" I mean a host compiler that implements the C++26 reflection wording; Clang has had an experimental implementation for some time.

## A generated driver is the wrong use.

I do not want the device compiler to reflect a peripheral and synthesize a driver. The rules of a real peripheral are full of exceptions, write-1-to-clear bits, and unlock sequences, and a generated driver that is almost right is worse than a short one that a person wrote and understands. I also do not want reflection near an interrupt. It is a compile-time facility, so there is nothing for it to compute on the core, and any version of the idea that suggests otherwise has misunderstood it.

What is left is the question of whether two descriptions still agree, and that is a question about the program's own types: does this struct still have the members the table says it has, at the sizes and offsets the table assumes? Compile-time introspection exists to answer questions of that kind, and the answer can be computed on the machine that builds the firmware, where the compiler is allowed to be newer than the one on the board.

## The host build goes red when the names disagree.

The table below pairs each register name with a mask of the bits the manual allows to be set. The masks are illustrative and not transcribed from RM0090; a real table would be checked against the manual by a person, once, and the reflection check would then keep the struct honest against that table.

```cpp
struct FieldSpec {
    std::string_view name;
    std::uint32_t legal;  // illustrative masks, not a paste from a manual
};

struct TimerRegs {
    std::uint32_t cr1;
    std::uint32_t dier;
    std::uint32_t sr;
    std::uint32_t arr;
    std::uint32_t ccr1;
};

inline constexpr FieldSpec kTimer[] = {
    {"cr1", 0x03FF},
    {"dier", 0x005F},
    {"sr", 0x001F},
    {"arr", 0xFFFF},
    {"ccr1", 0xFFFF},
};

consteval bool datasheet_matches() {
    std::size_t expect = 0;
    std::size_t seen = 0;
    for (auto mem : std::meta::nonstatic_data_members_of(^^TimerRegs)) {
        auto name = std::meta::identifier_of(mem);
        bool found = false;
        for (auto spec : kTimer) {
            if (spec.name == name) {
                found = true;
            }
        }
        if (!found || std::meta::offset_of(mem) != expect || std::meta::size_of(mem) != 4) {
            return false;
        }
        expect += 4;
        ++seen;
    }
    return seen == std::size(kTimer);
}

static_assert(datasheet_matches());
```

The snippet is a sketch written against the P2996 wording, and I have not built it with a specific compiler.

The check fails for three reasons, and each corresponds to a way the two descriptions drift. A member of `TimerRegs` with no row in `kTimer` is a register the mock knows nothing about, which is exactly the `CCR2` case, so the function returns false. A row that names a member which does not exist leaves the member count short of the table size, and the final comparison fails. And because the loop also requires the offsets to be 0, 4, 8, and so on, each with a size of 4 bytes, it catches a surprise padding hole and a field that someone typed as `uint16_t`. The name of each member comes back from `identifier_of` as a `std::string_view`, which is why it can be compared directly with the name in the table.

The function is `consteval`, so the list of members that `nonstatic_data_members_of` returns never exists at runtime. It lives and dies during constant evaluation, and the only interface the user sees is the `static_assert`. When the struct and the table disagree, the host build stops, and that is the whole behaviour I want from it.

## The mock reads the same table.

```cpp
class ShadowTimer {
 public:
    bool write(std::string_view name, std::uint32_t value) {
        for (std::size_t i = 0; i < std::size(kTimer); ++i) {
            if (kTimer[i].name != name) {
                continue;
            }
            if ((value & ~kTimer[i].legal) != 0) {
                reserved_write_ = true;
                return false;
            }
            words_[i] = value;
            touched_ |= 1u << i;
            return true;
        }
        return false;
    }

    std::uint32_t touched() const { return touched_; }
    bool reserved_write() const { return reserved_write_; }

 private:
    std::uint32_t words_[std::size(kTimer)]{};
    std::uint32_t touched_ = 0;
    bool reserved_write_ = false;
};
```

The mock indexes the same `kTimer` that the check validates, so it cannot hold a name the struct does not have. Its `write()` rejects a value that sets a bit outside the mask and records which names were touched. A bootloader test can then say more than "it returned true". It can say that after the bank switch the touch set contains only the one register the switch is supposed to use, and I prefer that to a green run that proves nothing about what was written.

The touch set is only as good as the calls the test makes. If no test exercises the path that writes the new field, the set will not show it. The mock narrows the gap between a test and the manual; it does not close it.

<figure>
  <img src="/images/reflection-datasheet.svg" alt="A host build reflects TimerRegs, compares it to a legal-bit table, and fails a static assertion on drift. The same table feeds a shadow mock. The device compiler receives an ordinary header.">
  <figcaption>Reflection runs on the host. The device compiler sees a header. The static assertion is what keeps the mock's names aligned with the struct.</figcaption>
</figure>

The snippet above does not print a header. An overlay of generated offsets for the device is optional, and it would be the same walk with an output stream attached, but I have not shipped a generator. The arrow to `timer_regs.h` can equally stand for the struct itself, maintained by hand and only guarded by the check. The valuable half is the `static_assert`, because it ties the mock's table to the type the firmware includes.

## Reflection earns its keep where a datasheet can lie.

This is a niche and not a style. Application structs have no datasheet, and reflecting them into a serializer is fine, but it is a different problem. Reflection earns its place where an external document and a second implementation, here the mock, must name the same fields, and where getting it wrong means a reserved-bit write instead of a messy log line. A register map is that situation.

Interrupt enumerations are a smaller instance of the same pattern. The enum of IRQ numbers and the table of handlers should be checked for the same set of names, and I would use the same host-side walk to do it. I would not start by generating the vector table.

The compile-time cost lands on the host. The device build does not get slower as long as it only includes the struct and never sees the reflection code.

## A matching name can still carry a wrong mask.

Reflection cannot read the PDF of the reference manual, and someone still types `0x03FF`. If that number is wrong but sits next to the right name, the check passes and the mock enforces the wrong rule. The job of the check is smaller than that: it makes sure that six months later the struct and the number still refer to the same identifier, and that the layout did not grow a hole.

C++20 concepts let me compile a driver twice against two models of an interface, and this use of reflection checks that the data those models talk about is still the same data. Neither belongs in the control loop.
