ColorEx
A python3 script

Version: 1.0.1
Release date: 2019.5.23

Requirements
    haishoku
    numpy
    pandas
    matplotlib

Usage
    1. run Solver.py.
    2. type in a path of a picture folder, and all results will save to 'results/' folder.
    3. if you want to adjust the plots, feel free to use plot_indie.py, a simple independent script for plotting.
    4. run jupyter notebook or ipython notebook, and open HSV_plotly.ipynb for 3D interative plotting. Edit the parameters yourself. Please do have fun ;)

Features
    1. get the dominant color and palette of all PNG files in the folder.
    2. output a picture of the dominant color and a picture of the palette of each original file.
    Each palette contains at most 8 colors sorted by percentage。
    3. output 2 csv files, including:
        1. dominant_array.csv, i.e. dominant color data of all files, including weight (times the color serves as a dominant one), RGB, HSV, Cartesian coordinates of HSV, and HEX.
        2. palette_array.csv, i.e. palette data of all files, including pic (file index), level (indicating relatize size of percentage), RGB, HEX, HSV, and Cartesian coordinates of HSV.
    4. output 4 scatterplots of all dominant colors displayed as both cylinder and circular corn from 2 angles.
    5. count program running time.
    6. plot_indie is an independent script for plotting. You may adjust your scatterplot using this script. Main parameters: s means size of the scatter points, alpha means the transparency of the scatter points.
    7. HSV_plotly.ipynb is an independent notebook for 3D interactive plotting.

Note
    1. appropriate values of parameters of the scatterplot depends on size and distribution characteristics of your data. Therefor, I have written plot_indie.py as an independent script for plotting, have fun!
    2. the Cartesian coordinates of HSV contained in the csv files are rounded to three decimal places, thus they are suitable for plotly to make 3D interactive plots, because it can makes the browser load faster.
