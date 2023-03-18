
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\saada\Downloads\PythonCode\Resized UI\build\assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("2560x1440")
window.configure(bg = "#1F1C25")


canvas = Canvas(
    window,
    bg = "#1F1C25",
    height = 1440,
    width = 2560,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    441.3721923828125,
    134.23330688476562,
    image=image_image_1
)

canvas.create_rectangle(
    0,
    15,
    2560,
    102,
    fill="#383442",
    outline="")

canvas.create_text(
    410,
    31,
    anchor="nw",
    text="Hyperloop Pod Vision Sensor System Panel",
    fill="#FFFFFF",
    font=("Alata Regular", 44 * -1)
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    441.3721923828125,
    477.73895263671875,
    image=image_image_2
)

canvas.create_rectangle(
    19.3721923828125,
    156.73895263671875,
    864.70556640625,
    790.7388916015625,
    fill="#383443",
    outline="")

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    1062.794189453125,
    134.23330688476562,
    image=image_image_3
)

canvas.create_text(
    900,
    124,
    anchor="nw",
    text="Objects Detected",
    fill="#FFFFFF",
    font=("Commissioner Bold", 21 * -1)
)

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    1062.794189453125,
    477.73895263671875,
    image=image_image_4
)

canvas.create_rectangle(
    878.794189453125,
    156.73895263671875,
    1248.62744140625,
    790.7388916015625,
    fill="#383442",
    outline="")

canvas.create_rectangle(
    1236.0,
    167.24444580078125,
    1242.0,
    779.2444458007812,
    fill="#2D2936",
    outline="")

canvas.create_text(
    63.39996337890625,
    124,
    anchor="nw",
    text="Live Video Feed",
    fill="#FFFFFF",
    font=("Commissioner Bold", 21 * -1)
)

image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    40.699951171875,
    134.03887939453125,
    image=image_image_5
)

image_image_6 = PhotoImage(
    file=relative_to_assets("image_6.png"))
image_6 = canvas.create_image(
    632.7250366210938,
    846.99169921875,
    image=image_image_6
)

canvas.create_text(
    180.513916015625,
    823.3195190429688,
    anchor="nw",
    text="Status: On Track Towards Destination",
    fill="#FFFFFF",
    font=("Commissioner Bold", 39 * -1)
)
window.resizable(True, True)
window.mainloop()
