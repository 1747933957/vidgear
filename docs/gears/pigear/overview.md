<!--
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019-2020 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
===============================================
-->

# PiGear API 

<figure>
  <img src="../../../assets/images/picam2.webp" alt="Pi Zero with Camera Module" loading="lazy" width="70%" />
  <figcaption>Raspberry Pi Camera Module</figcaption>
</figure>

## Overview

> PiGear is similar to [CamGear API](../../camgear/overview/) but exclusively made to support various Raspberry Pi Camera Modules _(such as OmniVision OV5647 Camera Module and Sony IMX219 Camera Module)_.

PiGear provides a flexible multi-threaded wrapper around complete [picamera](https://picamera.readthedocs.io/en/release-1.13/index.html) python library, and also provides us the ability to exploit almost all of its parameters like _brightness, saturation, sensor_mode, iso, exposure, etc._ effortlessly. Furthermore, PiGear supports multiple camera modules, such as in case of Raspberry Pi Compute module IO boards.

Best of all, PiGear provides excellent error-handling with features like a ==Threaded Internal Timer== - that keeps active track of any frozen-threads/hardware-failures robustly, and exit safely if it does occurs, i.e. If you're running PiGear API in your script, and someone accidentally pulls Camera module cable out, instead of going into possible kernel panic, PiGear will exit safely to save resources. 

!!! tip "Helpful Tips"

	* If you're already familar with [OpenCV](https://github.com/opencv/opencv) library, then see [Switching from OpenCV ➶](../../switch_from_cv/#switching-videocapture-apis)

	* It is advised to enable logging(`logging = True`) on the first run for easily identifying any runtime errors.

&thinsp; 

## Importing

You can import PiGear API in your program as follows:

```python
from vidgear.gears import PiGear
```

&thinsp;

## Usage Examples

<div class="zoom">
<a href="../usage/">See here 🚀</a>
</div>


## Parameters

<div class="zoom">
<a href="../params/">See here 🚀</a>
</div>

## Reference

<div class="zoom">
<a href="../../../bonus/reference/pigear/">See here 🚀</a>
</div>


## FAQs

<div class="zoom">
<a href="../../../help/pigear_faqs/">See here 🚀</a>
</div>  


&thinsp;