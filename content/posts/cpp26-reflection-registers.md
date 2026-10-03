---
title: "The mock and the register struct drifted"
date: 2025-06-22
draft: false
categories: ["Embedded", "C++"]
tags: ["C++26", "reflection", "registers", "testing"]
summary: "While compiling a UART driver twice, I noticed the host mock and the register struct could drift, and a host-side C++26 reflection check is enough to make that a build error."
description: "A short note on using C++26 reflection on the host to keep an embedded register map and its test mock on the same fields."
---

I have been writing a UART driver that compiles twice. Once against a host mock, once against the real memory-mapped registers. The driver is written with C++20 concepts, so the hot path is a template and not a vtable. The host build runs the tests. The device build goes in the flash image.

While doing that, I added a compare register to a timer struct that uses the same HAL style. The device header grew a field. The mock's table of legal bits did not. The tests stayed green, because none of them named the new field.

On the bench, that turns into one of two things. Either I write a reserved bit, or I use a register that no test ever touches. Both are quiet failures. I only found this one because I happened to open both files in the same afternoon.

## What I wanted from C++26

What I wanted is small, and it lives on the host. The vendor toolchain for the board lags behind. The host compiler can be new.

Reflection gives me the prefix `^^` and a splice `[: refl :]`, plus queries for the nonstatic data members of a type, a member's name, its offset, and its size. I use those in one `consteval` function and one `static_assert`. The device compiler never sees that code. It keeps compiling ordinary C++20 and includes the plain struct.

Here is the check. The masks are illustrative, not copied from a reference manual. I have not built this exact snippet on a specific compiler, so treat it as a sketch of the shape.

```cpp
struct FieldSpec {
    std::string_view name;
    std::uint32_t legal;
};

struct TimerRegs {
    std::uint32_t cr1;
    std::uint32_t dier;
    std::uint32_t sr;
    std::uint32_t arr;
    std::uint32_t ccr1;
};

inline constexpr FieldSpec kTimer[] = {
    {"cr1", 0x03FF}, {"dier", 0x005F}, {"sr", 0x001F},
    {"arr", 0xFFFF}, {"ccr1", 0xFFFF},
};

consteval bool datasheet_matches() {
    std::size_t expect = 0;
    std::size_t seen = 0;
    for (auto mem : std::meta::nonstatic_data_members_of(^^TimerRegs)) {
        auto name = std::meta::identifier_of(mem);
        bool found = false;
        for (auto spec : kTimer)
            if (spec.name == name) found = true;
        if (!found || std::meta::offset_of(mem) != expect || std::meta::size_of(mem) != 4)
            return false;
        expect += 4;
        ++seen;
    }
    return seen == std::size(kTimer);
}

static_assert(datasheet_matches());
```

It catches three things. A new member with no row in the table fails. A row in the table whose member is gone fails, because the counts no longer match. A padding hole, or a `uint16_t` where I assumed a word, fails on the offset or size check.

The mock indexes the same table. A write that sets a bit outside the mask fails the test, so the table is something the tests lean on and not a list I keep by hand.

## The driver stays handwritten

I do not want reflection to generate the driver. Peripheral rules have write-1-to-clear bits and unlock sequences. A generated driver that is almost right is worse than a short one I can read. There is also nothing for reflection to do at interrupt time. It is all compile-time.

## Where it helps

It helps on embedded because the check runs where the compiler is allowed to be newer than the one that builds the flash image. The image does not get bigger. Nothing changes on the device.

The same pattern works for an IRQ enum and the handler table. That is the smaller case, and the one I would reach for next.

It has limits. Reflection cannot read the PDF of the manual. A wrong mask next to the right name passes. All the check does is keep the struct, the offsets, and the names tied together six months from now, when I have forgotten why `ccr1` is there.

Concepts let the driver compile against two types. This check is what tells me those two types still talk about the same registers.
