# riibiitCam
A quadralens camera that creates wiggle style photography
The riibiit Camera

## Introduction
The riibiit Camera is stereoscopic lenticular quadra lens designed to add a touch of nostalgia to modern day digital photography. Popularized by gimmick cameras of the 1980’s such as the Nimslo, and Nishikia N8000, riibiit Camera aims to revive the charm of wiggle photography in a digital era. Utilizing the Kaeru effect, the images taken on this camera will hop to the next like a frog on a lily pad.

Confused? Don’t be, what this really boils down to is a camera with four lenses all equidistant from one another, capturing the same subject, at the same time and animating between each frame. 

Until now, this has been an effort only created in film photography as the modern digital camera sensor is simply not large enough to house four glass units. While many popular photography companies such as Canon and Sony have built 3D cameras using two glass units, or even in some cases three, no company has yet to fully realize the quadra lens system. Is it impossible? Well, at this time yes, the cost to create something like this would be in the millions for just R&D alone -- so how is some random nerd going to make this possible?


## Features
Hardware Setup: Utilizing the Raspberry Pi 4B and the Arducam 4-camera array hat, the riibiit Camera captures four near-identical photos arranged equidistantly from one another.

Digital Wiggle-Photography: Leveraging Python and libraries like Pillow and ImageMagick, the application seamlessly transforms the set of four images into a dynamic .gif. This creates a mesmerizing animation, reminiscent of the stereoscopic film cameras of the 1980s.

Filling a Void: While specific EOL traditional film cameras are able to achieve effect, the riibiit Camera steps in as a digital alternative. Unlike the constraints faced by big players like Canon and Sony, this project combines four lenses on one unit, offering a cost-effective and fully digital solution.


Wiggle Photography: Wiggle Photography, a dynamic technique seamlessly transitioning between frames, has seen a resurgence on platforms like Instagram. Initially dismissed as a gimmick, its charm now dominates visual storytelling, captivating audiences with playful, eye-catching loops. Brands like RETO have contributed to its revival, reaffirming wiggle photography as a captivating form of expression in the evolving digital landscape.



## Minimum Viable Product
The riibiit Camera should be able to…
Snap an image using all four lenses
Crop this image into 4 separate jpegs
Convert the jpegs into a seamless looping .gif


## Stretch Goals

Intelligent Image Processing
Incorporating something like the OpenCV library to detect objects/faces in the images, the riibiit Camera will use the tracking information to help center and align the subjects in the gif for a smoother loop.

Cloud-Saving with React Front End
 Creating a React-based front end (phone app counterpart?) that seamlessly saves the images to the cloud. This feature ensures that users can download their images and post them to the social media of their choosing.

Buttons
Using hardware buttons to load the program off the pi and snap the image upon click.

Camera mode only
The pi should boot only as a camera and not with the linux os.

