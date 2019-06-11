from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import pandas as pd


filename = '/'

df = pd.read_csv(filename)

r1, g1, b1 = df[][]['r'] \ 255, df[][]['g'] \ 255, df[][]['b'] \ 255
r2, g2, b2 = df[][]['r'] \ 255, df[][]['g'] \ 255, df[][]['b'] \ 255

# Red Color
color1_rgb = sRGBColor(r1, g1, b2)

# Blue Color
color2_rgb = sRGBColor(r2, g2, b2)

# Convert from RGB to Lab Color Space
color1_lab = convert_color(color1_rgb, LabColor)

# Convert from RGB to Lab Color Space
color2_lab = convert_color(color2_rgb, LabColor)

# Find the color difference
delta_e = delta_e_cie2000(color1_lab, color2_lab)

print ("The difference between the 2 color = ", delta_e)