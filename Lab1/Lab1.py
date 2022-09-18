#!/usr/bin/python3

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from math import isnan


# Definition of various types of derivative calculation methods
def deriv1(f, x, h):
    return (f(x + h) - f(x)) / h


def deriv2(f, x, h):
    return (f(x) - f(x - h)) / h


def deriv3(f, x, h):
    return (f(x + h) - f(x - h)) / 2 / h


def deriv4(f, x, h):
    return (4 / 3 * (f(x + h) - f(x - h)) / 2 / h - 1 / 3 * (f(x + 2 * h) - f(x - 2 * h)) / 4 / h)


def deriv5(f, x, h):
    return (3 / 2 * (f(x + h) - f(x - h)) / 2 / h - 3 / 5 * (f(x + 2 * h) - f(x - 2 * h)) / 4 / h + 1 / 10 * (f(x + 3 * h) - f(x - 3 * h)) / 6 / h)



# Definition of considered functions
def f1(x):
    return np.sin(x ** 2)


def f2(x):
    return np.cos(np.sin(x))


def f3(x):
    return np.exp(np.sin(np.cos(x)))


def f4(x):
    return np.log(x + 3.0)


def f5(x):
    return (x + 3.0) ** 0.5



# Definition of analytic derivatives of considered functions
def f1_deriv(x):
    return 2 * x * np.cos(x ** 2)


def f2_deriv(x):
    return -np.sin(np.sin(x)) * np.sin(x)


def f3_deriv(x):
    return f3(x) * np.cos(np.cos(x)) * (-np.sin(x))


def f4_deriv(x):
    return 1.0 / (x + 3.0)


def f5_deriv(x):
    return 1.0 / 2.0 / np.sqrt(x + 3.0)



# Store this whole stuff into arrays to properly work with it 
functions_array      = np.array([f1,        f2,       f3,       f4,       f5])
derivatives_array    = np.array([f1_deriv, f2_deriv, f3_deriv, f4_deriv, f5_deriv])
deriv_type_array     = np.array([deriv1,    deriv2,   deriv3,   deriv4,   deriv5])
function_names_array = np.array(["$\sin(x^2)$", "$\cos(\sin(x))$", "$e^{\sin(\cos(x))}$", "$\ln(x + 3)$", "$(x + 3)^{0.5}$"])
colors_array = np.array(["red", "green", "blue", "black", "magenta"])

# Current ax (function graph) index: 0, 1, 2, 3, 4
curr_ax_idx = 0

# Flag indicating whether scale is linear
isLinear = False

# Flag indicating whether images should be saved. False by default
save = False



def main():

    # Arrange h-steps
    n_arr = np.array([n for n in range(1, 22)])
    h_arr = 2.0 / (2.0 ** n_arr)
    
    # Initial point where we start approximating a derivative
    x_0 = 10.0

    # Try to create a folder to store the pictures in case we save them
    if save:
        try:
            os.mkdir("./Pictures")
        except:
            pass
    
    # Now let's define and configure a figure for plots
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.set_window_title('Functions')

    # A couple of lists to store information about buttons, axes and plots themselves
    buttons = []
    axes = []
    plots_plots = []
    
    # Define and place buttons in correct location
    for i in range(0, len(functions_array)):
        ax_button = plt.axes([0.906, 0.8 - i * 0.06, 0.088, 0.05])
        buttons.append(Button(ax_button, function_names_array[i], color='gold', hovercolor='skyblue'))

    # The same, but for a <Scale> button
    ax_scale_button = plt.axes([0.906, 0.2, 0.088, 0.05])
    buttons.append(Button(ax_scale_button, "Scale: Log/Lin", color='gold', hovercolor='skyblue'))

    # Creating a slider to change a point where we approximate the derivative
    ax_slide = plt.axes([0.15, 0.95, 0.7, 0.03])
    slider = Slider(ax_slide, "$x_0$", valmin=-2.9, valmax=100.0, valinit=x_0, valstep=0.01)


    # Configuring the axes. Single plot is just a curve for a single method of approximation
    # We have 5 functions and 5 methods for all of them - 25 in total
    for i in range(0, len(functions_array)):
        axes.append(fig.subplots())
        plots = [axes[i].plot([], []) for n in range(0, len(functions_array))]
        plots_plots.append(plots)

        axes[i].set_title(f"Absolute value of error of derivative approximation of {function_names_array[i]} at point $x_0 = {x_0}$", fontsize=18)
        axes[i].grid(which='major', color="black", linewidth="1.7")
        axes[i].grid(which='minor', color="grey", linewidth='0.2')
        axes[i].tick_params(axis="x", labelsize=20)
        axes[i].tick_params(axis="y", labelsize=20)
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_xlabel("$\log_{10} \; h_n$", fontsize=20)
        axes[i].set_ylabel("$\log_{10} \; \epsilon$", fontsize=20)
        axes[i].set_visible(False)

    # The main filling loop
    for fig_num in range(0, len(functions_array)):
        try:
            func  = functions_array[fig_num]
            deriv = derivatives_array[fig_num]

            for n in range(0, len(functions_array)):
                deriv_type = deriv_type_array[n]

                # We obtain an absolute error of approximation
                y_arr = np.abs(deriv(x_0) - deriv_type(func, x_0, h_arr))
                
                # Plot it. Just a single curve, repeat 5 times in total for a single ax (function)
                plots_plots[fig_num][n], = axes[fig_num].plot(h_arr, y_arr, color=colors_array[n], marker="o", linewidth='2.5')
                plots_plots[fig_num][n].set_label(f"Method {n + 1}")
            
            # Only now we can set the legend, because we have not had information about curves before 
            axes[fig_num].legend(fontsize=15, loc='lower right')
        
            # Save figures if we need it
            if save:
                fig.savefig(f"./Pictures/Function_{fig_num + 1}")

        except KeyboardInterrupt:
            return 0


    # Update callback to rebuild all if the slider was moved or a button was pressed
    def update(val):
        global curr_ax_idx
        func  = functions_array[curr_ax_idx]
        deriv = derivatives_array[curr_ax_idx]
        x_0 = slider.val

        min_val, max_val = axes[curr_ax_idx].get_ylim()

        # The same calculating loop as before
        for n in range(0, len(functions_array)):
            deriv_type = deriv_type_array[n]
        
            y_arr = np.abs(deriv(x_0) - deriv_type(func, x_0, h_arr))
            
            plots_plots[curr_ax_idx][n].set_xdata(h_arr)
            plots_plots[curr_ax_idx][n].set_ydata(y_arr)
            
            min_val = min(np.min(y_arr), min_val)
            max_val = max(np.max(y_arr), max_val)
            
        # Try to set proper limits for axis, but only if they are actual numbers, not NaN-s
        if not (isnan(min_val) or isnan(max_val)):
            axes[curr_ax_idx].set_ylim([min_val, max_val])
        
        # Update the title
        axes[curr_ax_idx].set_title(f"Absolute value of error of derivative approximation of {function_names_array[curr_ax_idx]} at point $x_0 = {round(x_0, 2)}$", fontsize=18)


    # Defining callbacks for event handling
    def switch_to_fun0(event):
        for ax in axes:
            ax.set_visible(False)

        global curr_ax_idx
        curr_ax_idx = 0
        update(0)
        axes[curr_ax_idx].set_visible(True)


    def switch_to_fun1(event):
        for ax in axes:
            ax.set_visible(False)
        
        global curr_ax_idx
        curr_ax_idx = 1
        update(0)
        axes[curr_ax_idx].set_visible(True)

    def switch_to_fun2(event):
        for ax in axes:
            ax.set_visible(False)
        
        global curr_ax_idx
        curr_ax_idx = 2
        update(0)
        axes[curr_ax_idx].set_visible(True)

    def switch_to_fun3(event):
        for ax in axes:
            ax.set_visible(False)
        
        global curr_ax_idx
        curr_ax_idx = 3
        update(0)
        axes[curr_ax_idx].set_visible(True)

    def switch_to_fun4(event):
        for ax in axes:
            ax.set_visible(False)
        
        global curr_ax_idx
        curr_ax_idx = 4
        update(0)
        axes[curr_ax_idx].set_visible(True)


    def set_scale(isLin):
        for i in range(0, len(axes)):
            axes[i].set_xscale("linear"     if isLin else 'log')
            axes[i].set_yscale("linear"     if isLin else 'log')
            axes[i].set_xlabel("$h_n$"      if isLin else "$\log_{10} \; h_n$", fontsize=20)
            axes[i].set_ylabel("$\epsilon$" if isLin else "$\log_{10} \; \epsilon$", fontsize=20)
            axes[i].autoscale()
    
    def change_scale(event):
        global isLinear
        isLinear = not isLinear
        set_scale(isLinear)
        update(0)

    # Registering all callbacks
    buttons[0].on_clicked(switch_to_fun0)
    buttons[1].on_clicked(switch_to_fun1)
    buttons[2].on_clicked(switch_to_fun2)
    buttons[3].on_clicked(switch_to_fun3)
    buttons[4].on_clicked(switch_to_fun4)

    buttons[5].on_clicked(change_scale)

    slider.on_changed(update)

    # Show current ax and show whole plot
    axes[curr_ax_idx].set_visible(True)
    plt.show()



if __name__ == '__main__':
    main()