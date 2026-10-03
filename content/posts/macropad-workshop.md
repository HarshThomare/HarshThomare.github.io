---
title: "IEEE@UCIxKeebs Macropad Workshop"
date: 2025-02-16
draft: false
categories: ["Workshops", "Hardware Design", "Electronics"]
tags: ["PCB Design", "KiCad", "Soldering", "QMK", "Mechanical Keyboards", "Arduino", "3D Printing"]
description: "A look back at leading the IEEE@UCIxKeebs Macropad Workshop, where 60 participants learned PCB design, hardware assembly, and QMK firmware configuration from scratch."
---
As the lead for this project, I had the incredible opportunity to organize and run the IEEE@UCIxKeebs Macropad Workshop. It was a phenomenal experience bringing together a diverse group of 60 participants, many of whom were learning PCB design, hardware assembly, and firmware programming for the very first time.

### Overview of the Workshop
The workshop was structured to provide a hands-on introduction to the world of custom keyboards and electronics. We split the event into three one-hour sessions, each with dedicated mentor guidance and access to our 25 soldering stations, so every participant got hands-on support throughout the build. By the end of the workshop, each participant had completed the full pipeline of hardware creation: from schematic routing to a fully functioning macropad.

![macropad](/images/KEEBS_CLUB_AT_UCI_X_IEEE_-_Macropad.jpg)

### Inspiration
The inspiration behind this project was to demystify hardware design and make it highly accessible to beginners. Custom macropads and keyboards serve as a fantastic gateway into engineering because they combine electrical engineering, mechanical design, and software programming into one satisfying, tangible product. We wanted to give students a project they could proudly use on their desks every day while gaining confidence in industry-standard skills like PCB routing and QMK firmware configuration.

### How We Built It
We broke the creation of the macropad down into three main instructional phases:

1. **Keyboard Design (Workshop 1):** We started with the schematic and general layout, where participants learned to place components and finalize their PCB designs using KiCad, an industry-standard tool for electronic design automation (EDA).
2. **3D CAD Enclosure (Workshop 2):** Participants explored mechanical design by building 3D CAD models for an enclosure tailored to their macropad.
3. **Assembly (Workshop 3):** This was the hands-on manufacturing phase. Students learned proper safety practices and soldering technique to attach the diode array, resistors, microcontroller pin headers, and key switches to the bare PCB. Finally, we guided them through flashing QMK firmware onto their controllers so the macropads could interface with their computers.

![Macropad](/images/macropad.jpg)

### Bill of Materials
Each macropad build required the following parts:
* Custom PCB
* Arduino Pro Micro + headers
* Micro USB cable
* Key switches
* Keycaps
* Diodes
* Resistors
* 3D printed case

### Design Files & Resources
If you're interested in recreating this project, viewing the schematics, or printing your own enclosure, you can access the files below:

[Download 3D Print Files (STL)](/assets/Macropad%20STL.stl)
<a href="https://docs.google.com/presentation/d/1RomjoYs0pbLGqU3FmWz12G2fLHtqptC8CVIVSVzKHPw/edit?usp=sharing" download>Download Workshop Slides</a>

You can also clone the full design repository on [GitHub](https://github.com/kevinthai-kbd/zotzotzotpad).
