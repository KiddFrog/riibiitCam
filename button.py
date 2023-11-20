from gpiozero import Button
from signal import pause
from time import sleep

button = Button(21)
x = 0
print("x currently is 0")

def press():
    global x
    x += 1
    print("x currently is " + str(x))

button.when_pressed = press

pause()
