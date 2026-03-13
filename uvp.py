#    URVA Visualization Program (UVP), version 0.1.0 
#    Copyright (C) 2026 Computational and Theory Chemistry Group (CATCO), Southern Methodist University, Dallas, TX, USA
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>


import os
import io
import sys 
import subprocess

import threading
from threading import Thread

import matplotlib
import darkdetect
import trace
import shutil
import stat
import wx
import fsleyes_widgets.floatslider
import numpy as np
import pandas as pd
import math
import time
import glob
import re

import wxmplot
from wxmplot import *

import ase
from ase.io.utils import PlottingVariables, make_patch_list
from ase.utils import writer
from ase.visualize import view

import PIL
from PIL import Image

# Check to see if dark mode is enabled by the user on their machine
current_mode = darkdetect.theme()
if current_mode == "Dark":
    print("Error: Dark mode has been detected on your machine. To improve the visibility of the UVP GUI widgets, please use light mode instead.")
    sys.exit()

# Print UVP initialization statement
if os.path.exists('./frames') == False:
    print("Initializing UVP version 0.1.0 \nGeneration and processing of animation frames may take a few minutes (one-time process). Please wait ... \n")
else:
    print("Initializing UVP version 0.1.0 \nPlease wait ... \n")

# Code for thread_with_trace (kills treads with trace exceptions) class comes from the following Stack Overflow discussion: https://stackoverflow.com/a/63264889
class thread_with_trace(threading.Thread):
    def __init__(self, *args, **keywords):
        threading.Thread.__init__(self, *args, **keywords)
        self.killed = False
 
    def start(self):
        self.__run_backup = self.run
        self.run = self.__run      
        threading.Thread.start(self)
 
    def __run(self):
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup
 
    def globaltrace(self, frame, event, arg):
        if event == 'call':
            return self.localtrace
        else:
            return None
 
    def localtrace(self, frame, event, arg):
        if self.killed:
            if event == 'line':
                raise SystemExit()
        return self.localtrace

    def kill(self):
        self.killed = True
        
# Modification of the ase.io.eps source code (https://wiki.fysik.dtu.dk/ase/_modules/ase/io/eps.html#write_eps). Date: 3.11.26
class EPS(PlottingVariables):
    def __init__(self, atoms, 
                 width, height,
                 rotation='', radii=None,
                 bbox=None, colors=None, scale=20, maxwidth=500,
                 **kwargs):
        """Encapsulated PostScript writer.

        show_unit_cell: int
            0: Don't show unit cell (default).  1: Show unit cell.
            2: Show unit cell and make sure all of it is visible.
        """
        PlottingVariables.__init__(
            self, atoms, rotation=rotation,
            radii=radii, bbox=bbox, colors=colors, scale=scale,
            maxwidth=maxwidth,
            **kwargs)
        
        self.mod_width = width + 50
        self.mod_height = height + 50

    def write(self, fd):
        renderer = self._renderer(fd)
        self.write_header(fd)
        self.write_body(fd, renderer)
        self.write_trailer(fd, renderer)

    def write_header(self, fd):
        fd.write('%!PS-Adobe-3.0 EPSF-3.0\n')
        fd.write('%%Creator: G2\n')
        fd.write('%%CreationDate: %s\n' % time.ctime(time.time()))
        fd.write('%%Orientation: portrait\n')
        bbox = (0, 0, self.mod_width, self.mod_height)
        fd.write('%%%%BoundingBox: %d %d %d %d\n' % bbox)
        fd.write('%%EndComments\n')

        Ndict = len(psDefs)
        fd.write('%%BeginProlog\n')
        fd.write('/mpldict %d dict def\n' % Ndict)
        fd.write('mpldict begin\n')
        for d in psDefs:
            d = d.strip()
            for line in d.split('\n'):
                fd.write(line.strip() + '\n')
        fd.write('%%EndProlog\n')

        fd.write('mpldict begin\n')
        fd.write('%d %d 0 0 clipbox\n' % (self.mod_width, self.mod_height))

    def _renderer(self, fd):
        # Subclass can override
        from matplotlib.backends.backend_ps import RendererPS
        return RendererPS(self.mod_width, self.mod_height, fd)

    def write_body(self, fd, renderer):
        patch_list = make_patch_list(self)
        for patch in patch_list:
            patch.draw(renderer)

    def write_trailer(self, fd, renderer):
        fd.write('end\n')
        fd.write('showpage\n')


def write_eps_mod(fd, atoms, width, height, **parameters):
    EPS(atoms, width, height, **parameters).write(fd)
    
# psDefs, the Python dictionary below, is adapted from Matplotlib 3.7.3
    
psDefs = [
    # name proc  *_d*  -
    # Note that this cannot be bound to /d, because when embedding a Type3 font
    # we may want to define a "d" glyph using "/d{...} d" which would locally
    # overwrite the definition.
    "/_d { bind def } bind def",
    # x y  *m*  -
    "/m { moveto } _d",
    # x y  *l*  -
    "/l { lineto } _d",
    # x y  *r*  -
    "/r { rlineto } _d",
    # x1 y1 x2 y2 x y *c*  -
    "/c { curveto } _d",
    # *cl*  -
    "/cl { closepath } _d",
    # *ce*  -
    "/ce { closepath eofill } _d",
    # w h x y  *box*  -
    """/box {
      m
      1 index 0 r
      0 exch r
      neg 0 r
      cl
    } _d""",
    # w h x y  *clipbox*  -
    """/clipbox {
      box
      clip
      newpath
    } _d""",
    # wx wy llx lly urx ury  *setcachedevice*  -
    "/sc { setcachedevice } _d",
]

# Creating and functionalizing the GUI frame

class MyFrame(wx.Frame):    
    def __init__(self):
        # Obtaining correct dimensions for program display
        super().__init__(parent=None, title='URVA Visualization Program',size=(1100,750))
        self.SetBackgroundColour("WHITE")
        
        # Create strings for file paths to curvature, energy, and movie files
        sys.stdout = sys.__stdout__
        self.currentDirectory = os.getcwd()
        if bool(glob.glob('*.dat')):
            self.cui_file_name = glob.glob('*.dat')[0]
            self.cui_file_path = self.currentDirectory + "/" + self.cui_file_name
        else:
            print("Error: RP curvature .dat file could not be found.")
            sys.exit()
        if bool(glob.glob('*.csv')):
            self.ene_file_name = glob.glob('*.csv')[0]
            self.ene_file_path = self.currentDirectory + "/" + self.ene_file_name
        else:
            print("Error: RC energy .csv file could not be found.")
            sys.exit()    
        if bool(glob.glob('*.xyz')):  
            self.mov_file_name = glob.glob('*.xyz')[0]
            self.mov_file_path = self.currentDirectory + "/" + self.mov_file_name
        else:
            print("Error: RC geometry .xyz file could not be found.")
            sys.exit()
        sys.stdout = io.StringIO()   
        
        # Processing of curvature input file
        self.s = []
        self.curvature = []
        dat_file = pd.read_csv(self.cui_file_path, sep=' ')
        dat_file.to_csv('originalkappa.csv', index=None)
        curvature_array = np.loadtxt('originalkappa.csv', delimiter=',', usecols=(0,1))
        os.remove("originalkappa.csv")
        for coord in curvature_array[:,0]:
            self.s.append(coord)
        for curv in curvature_array[:,1]:
            self.curvature.append(curv)
        self.curv_coord = np.array(self.s)
        self.curv_val = np.array(self.curvature)
        # Removal of duplicates (combinations of s and curvature values that share the same s value)
        original_length = len(self.curv_coord)
        for i in range(1,original_length):
            current_length = len(self.curv_coord)
            if i < current_length:
                if self.curv_coord[i] == self.curv_coord[i-1]:
                    self.curv_coord = np.delete(self.curv_coord, i-1)
                    self.curv_val = np.delete(self.curv_val, i-1)
        
        # Processing of energy input file
        self.energy_s = []
        self.energy = []
        energy_array = np.loadtxt(self.ene_file_path, skiprows=1)
        for coord in energy_array[:,0]:
            self.energy_s.append(coord)
        original_elec_energy = energy_array[:,1][0]
        for elec_energy in energy_array[:,1]:
            converted_energy = (elec_energy - original_elec_energy)*627.5096
            self.energy.append(converted_energy)
        self.energy_coord = np.array(self.energy_s)
        self.energy_val = np.array(self.energy)
        # Removal of duplicates (combinations of s and energy values that share the same s value)
        original_length = len(self.energy_coord)
        for j in range(1,original_length):
            current_length = len(self.energy_coord)
            if j < current_length:
                if self.energy_coord[j] == self.energy_coord[j-1]:
                    self.energy_coord = np.delete(self.energy_coord, j-1)
                    self.energy_val = np.delete(self.energy_val, j-1)        
        
        # Processing of movie input file
        self.xyz_lines = []                             
        with open (self.mov_file_path, 'rt') as xyz: 
            for line in xyz:                
                self.xyz_lines.append(line) 
        xyz.close()
        self.n_atoms = int(self.xyz_lines[0])
        
        self.cui_lines_u = []
        self.xyz_cui_line_nums = []
        for j in range(1,len(self.xyz_lines)):
            if (j+self.n_atoms+1) % (self.n_atoms+2) == 0:
                self.cui_lines_u.append(self.xyz_lines[j])
                self.xyz_cui_line_nums.append(j)
        
        self.cui_lines_f = []
        for cui_line in self.cui_lines_u:
            for c in range(len(cui_line)):
                if cui_line[c].isnumeric() or (cui_line[c] == '-' and cui_line[c+1].isnumeric()):
                    self.cui_lines_f.append(float("{:.{}f}".format(float(cui_line[c:len(cui_line)-2]), 5)))
                    break
        
        
        # A dictionary with the all curvature coordinate values and their corresponding line numbers in the 
        # list self.xyz_lines, which contains each line of the original movie file as a list element.
        self.cui_set = {self.xyz_cui_line_nums[l]: self.cui_lines_f[l] for l in range(len(self.xyz_cui_line_nums))}
        
        # Creates slider min and max variables, along with initial frame and frame rate.
        self.slider_min = np.min(self.s)
        self.slider_max = np.max(self.s)
        init_value = self.slider_min
        self.frame_rate = 25
        
        # Identifies first and last frames (corresponding to self.slider_min and self.slider_max) in the cui_set dictionary:
        cui_coords = list(self.cui_set.values())
        self.first_frame = 0
        self.last_frame = len(cui_coords)
        
        for cui_coord in cui_coords:
            min_distance = abs(cui_coord-self.slider_min)
            if min_distance < 0.02:
                first_target = cui_coord
                break
        self.first_frame = cui_coords.index(first_target)
        
        for cui_coord in cui_coords:
            max_distance = abs(cui_coord-self.slider_max)
            if max_distance < 0.02:
                last_target = cui_coord
                break
        self.last_frame = cui_coords.index(last_target)
        
        
        # Creates the "frames" directory, which contains the .png images of all the points along the reaction coordinate 
        # generated with ase package. If the "frames" directory is already present, this step is skipped.
    
        sys.stdout = sys.__stdout__
        if os.path.exists('./frames') == False:
            # Get dimensions for very first frame
            s_value = self.cui_set.get(1)
            xyz_name = "./" + str(s_value)+".xyz"       
            xyz_file = open(xyz_name, "w")
            for r in range(0,self.n_atoms+2):
                self.xyz_lines[r]
                xyz_file.write(self.xyz_lines[r])
            xyz_file.close()
            structure = ase.io.read(xyz_name)
            eps_path = "./" + str(s_value)+ ".eps"
            ase.io.write(eps_path,structure)
            first_frame_eps = Image.open(eps_path)
            self.w, self.h = first_frame_eps.size
            os.remove(eps_path)
            os.remove(xyz_name)
            # Create frames directory
            os.mkdir('frames')
            # Create xyz files (and their corresponding png files) for every value of s described in the movie file.
            frame_count = 1
            for line_value in self.cui_set.keys():
                s_value = self.cui_set.get(line_value)
                xyz_name = "./frames/"+str(s_value)+".xyz"       
                xyz_file = open(xyz_name, "w")
                for r in range(line_value-1,line_value+self.n_atoms+1):
                    self.xyz_lines[r]
                    xyz_file.write(self.xyz_lines[r])
                xyz_file.close()
                # Use ASE to generate .eps image corresponding to each xyz geometry
                structure = ase.io.read(xyz_name)
                eps_path = "./frames/"+str(s_value)+".eps"
                handler = open(eps_path,"x")
                write_eps_mod(handler,structure,self.w,self.h)
                handler.close()
                # Process the eps image and convert it into a high-resolution png image (with a white background) - Code adapted from Stack Overflow discussion: https://stackoverflow.com/a/60238673
                resolution = (1024,1024)
                vector = Image.open(eps_path)
                vector.load(scale=10)
                if vector.mode in ('P','1'):
                    vector = vector.convert("RGB")
                size_ratio = min(resolution[0]/vector.size[0], resolution[1]/vector.size[1])
                newW = int(vector.size[0]*size_ratio)
                newH = int(vector.size[1]*size_ratio)
                scaled_size = (newW,newH)
                vector = vector.resize(scaled_size,Image.LANCZOS)
                png_path = "./frames/"+str(s_value)+".png"
                vector.save(png_path, "PNG")
                # Remove the intermediary eps, and xyz files from the frames directory 
                os.remove(eps_path)
                os.remove(xyz_name)
                # Update user on the progress of the frame generation
                message_update = str(int((frame_count/len(self.cui_set.keys()))*100)) + "% of animation frames generated"
                print(message_update + '\r', sep='', end ='', file = sys.stdout , flush = False) # Iterative update of frame generation - solution acquired from Stack Overflow discussion: https://stackoverflow.com/a/60969051)
                frame_count += 1
        print("\n")
        # Processes images to assess how wide-spanning they are
        if os.path.exists('./frames/max_position.txt') == False:
            frames_directory = os.fsencode("./frames")
            column_mean_set = []
            filenum = 1
            for file in os.listdir(frames_directory):
                filename = "./frames/" + os.fsdecode(file)
                frame = Image.open(filename)
                for w in range(frame.width-1,0,-1):
                    no_color_right = True
                    for h in range(frame.height):
                        red, green, blue = frame.getpixel((w,h))
                        if red != 255 or green != 255 or blue != 255:
                            no_color_right = False
                            break
                    if no_color_right == False:
                        end_column = w
                        break
                for w in range(0,frame.width):
                    no_color_left = True
                    for h in range(frame.height):
                        red, green, blue = frame.getpixel((w,h))
                        if red != 255 or green != 255 or blue != 255:
                            no_color_left = False
                            break
                    if no_color_left == False:
                        start_column = w
                        break
                column_mean_set.append(np.average([start_column,end_column])/frame.width)
                message_update = str(int((filenum/len(os.listdir(frames_directory)))*100)) + "% of animation frames processed"
                print(message_update + '\r', sep='', end ='', file = sys.stdout , flush = False) # Iterative update of animation frame processing - solution acquired from Stack Overflow discussion: https://stackoverflow.com/a/60969051)
                filenum += 1
            max_column_mean = max(column_mean_set)
            max_position_file = open('./frames/max_position.txt','w')
            max_position_file.write(str(max_column_mean))
            max_position_file.close()                
        column_mean_file = open('./frames/max_position.txt')
        max_column_mean = float(column_mean_file.readlines()[0])
        sys.stdout = io.StringIO()    
        
        # Creates variables showing status of forward animation thread (tf) and reverse animation thread (tr)
        self.tf_alive = False
        self.tr_alive = False
        
        # Creates variables showing state of energy and curvature plot tracers (True if enabled and present, False if disabled and not present)
        self.tracers_enabled = True
        self.tracers_present = True
        
        # Creates variables showing state of slider circle 
        self.at_min = True
        self.at_max = False
        
        # Adjustment value definition (if the panels need to be shifted)
        self.a = -325
        
        # Original number of active running threads are enumerated here
        self.standby_thread_count = threading.active_count()
        
        # Creates the curavture plot panel and its sizer
        self.curv_plot_panel = wx.Panel(self,pos=(400+self.a,10),size=(300,300))
        self.curv_plot_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Creates the curvature plot (and its corresponding tracer)
        self.curv_plot = wxmplot.PlotPanel(self.curv_plot_panel, size=(300,300), dpi=100,fontsize=5)
        self.curv_plot.plot(self.curv_coord,self.curv_val,bgcolor = "WHITE",framecolor="WHITE",xlabel="Reaction parameter " + u's [amu¹\u141F² Bohr]',ylabel="Curvature",ymin=0,ymax=20)
        self.curv_tracer_x = np.array([self.slider_min,self.slider_min])
        self.curv_tracer_y = np.array([-(max(self.curv_val)+50),(max(self.curv_val)+50)])
        self.curv_plot.oplot(self.curv_tracer_x,self.curv_tracer_y)
        
        self.curv_plot_panel.SetSizer(self.curv_plot_sizer)
        self.curv_plot_panel.Layout()

        # Creates the energy plot panel and its sizer
        self.energy_plot_panel = wx.Panel(self,pos=(400+self.a,285),size=(300,300))
        self.energy_plot_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Creates the energy plot (and its corresponding tracer)
        self.energy_plot = wxmplot.PlotPanel(self.energy_plot_panel, size=(300,300), dpi=100,fontsize=5)
        self.energy_plot.plot(self.energy_coord,self.energy_val,bgcolor = "WHITE",framecolor="WHITE",xlabel="Reaction parameter " + u's [amu¹\u141F² Bohr]',ylabel="Energy (kcal/mol)",ymin=(min(self.energy_val)-5),ymax=max(self.energy_val)+5)
        self.energy_tracer_x = np.array([self.slider_min,self.slider_min])
        self.energy_tracer_y = np.array([(min(self.energy_val)-50),(max(self.energy_val)+50)])
        self.energy_plot.oplot(self.energy_tracer_x,self.energy_tracer_y)
        
        self.energy_plot_panel.SetSizer(self.energy_plot_sizer)
        self.energy_plot_panel.Layout()
        
        # Creates the scaled animation image (initial image)
        for line_value in self.cui_set.keys():
            s = self.cui_set.get(line_value)
            distance = abs(float(s)-self.slider_min)
            if distance < 0.0015:
                self.initial_s = s
                break
        self.initial_geo = "./frames/" + str(self.initial_s) + ".png"
        self.initial_image = wx.Image(self.initial_geo,wx.BITMAP_TYPE_ANY)
        aspect_ratio = self.initial_image.GetWidth()/self.initial_image.GetHeight()
        NewH = 400
        NewW = int(NewH * aspect_ratio) 
        self.initial_image = self.initial_image.Scale(NewW,NewH)
        
        # Creates the visualization panel and its sizer
        self.visual_panel = wx.Panel(self,pos=(int(760-(max_column_mean*NewW)),10),size=(800,600))
        self.visual_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Creates static bitmap of initial image
        self.img = wx.StaticBitmap(self.visual_panel, wx.ID_ANY,wx.Bitmap(self.initial_image))
        
        self.visual_panel.SetSizer(self.visual_sizer)
        self.visual_panel.Layout()
        
        # Creates the slider panel and its sizer
        self.slider_panel = wx.Panel(self,pos=(910+self.a,500),size=(350,300))
        self.slider_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Slider left label creation
        left_label_text = 's = ' + "{:.{}f}".format(self.slider_min,4)
        self.left_label = wx.StaticText(self.slider_panel,label = left_label_text)
        self.left_label.SetForegroundColour((255,0,0))
        self.slider_sizer.Add(self.left_label, 0, wx.ALL, 5)
    
        # Slider creation
        self.my_slider = fsleyes_widgets.floatslider.FloatSlider(self.slider_panel, value = init_value, minValue = self.slider_min,maxValue = self.slider_max) # Creates the slider widget
        self.my_slider.Bind(wx.EVT_SLIDER, self.on_slide) # Invokes the on_slide function if the slider is moved

        self.slider_sizer.Add(self.my_slider, 0, wx.ALL, 5) # Adds the slider to the sizer                   
        
        # Slider right label creation
        right_label_text = 's = ' + "{:.{}f}".format(self.slider_max,4)
        self.right_label = wx.StaticText(self.slider_panel,label = right_label_text)
        self.right_label.SetForegroundColour((255,0,0))
        self.slider_sizer.Add(self.right_label, 0, wx.ALL, 5)
        
        self.slider_panel.SetSizer(self.slider_sizer)
        self.slider_panel.Layout()
        
        # Creates the slider controller panel and its sizer
        self.slider_c_panel = wx.Panel(self,pos=(900+self.a,530),size=(1000,100))
        self.slider_c_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Number entry box left label creation
        self.num_entry_left_label = wx.StaticText(self.slider_c_panel,label = u's[amu¹\u141F² Bohr]:')
        self.slider_c_sizer.Add(self.num_entry_left_label, 0, wx.ALL, 5)
        
        # Number entry box creation
        self.text_ctrl = wx.TextCtrl(self.slider_c_panel, size=(100,-1), style = wx.TE_CENTRE)
        self.text_ctrl.SetValue("{:.{}f}".format(self.slider_min,4))
        self.text_ctrl.Bind(wx.EVT_KEY_DOWN, self.onButtonKeyEvent)
        self.slider_c_sizer.Add(self.text_ctrl, 0, wx.ALL, 5)
        
        # Jump button creation
        self.jump_btn = wx.Button(self.slider_c_panel, label='Jump to s value')
        self.jump_btn.Bind(wx.EVT_BUTTON, self.on_jump)
        self.slider_c_sizer.Add(self.jump_btn, 0, wx.ALL, 5)        
        
        self.slider_c_panel.SetSizer(self.slider_c_sizer)
        self.slider_c_panel.Layout()
        
        # Creates the slider frame rate panel and its sizer
        self.slider_f_panel = wx.Panel(self,pos=(450+self.a,580),size=(400,100))
        self.slider_f_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Frame rate text box creation
        self.fr_text_ctrl = wx.TextCtrl(self.slider_f_panel, size=(100,-1))        
        self.fr_text_ctrl.Bind(wx.EVT_KEY_DOWN, self.onButtonKeyEvent_fr)
        self.slider_f_sizer.Add(self.fr_text_ctrl, 0, wx.ALL, 5)
        
        # Frame rate button creation
        self.fr_btn = wx.Button(self.slider_f_panel, label='Set Frame Rate')
        self.fr_btn.Bind(wx.EVT_BUTTON, self.on_fr_btn)
        self.slider_f_sizer.Add(self.fr_btn, 0, wx.ALL, 5)
        
        self.slider_f_panel.SetSizer(self.slider_f_sizer)
        self.slider_f_panel.Layout()
        
        # Set the default frame rate in the frame rate text box upon starting the program
        self.fr_text_ctrl.SetValue(str(self.frame_rate))
        
        # Disable the frame rate text box by default
        self.fr_text_ctrl.Disable()
        
        # Creates the slider frame rate message + tracer disabler/enabler panel and its sizer
        self.slider_fm_panel = wx.Panel(self,pos=(450+self.a,610),size=(400,100))
        self.slider_fm_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Creates the default frame rate message below the fps entry box
        self.fps_message = wx.StaticText(self.slider_fm_panel,label = u'Default frame rate: 25 fps')
        self.fps_message.SetForegroundColour((255,0,0))
        self.slider_fm_sizer.Add(self.fps_message, 0, wx.ALL, 5)
        
        # Creates the current frame rate message below the default frame rate message 
        self.fps_message_current = wx.StaticText(self.slider_fm_panel,label = 'Current frame rate: ' + str(self.frame_rate) + ' fps')
        self.fps_message_current.SetForegroundColour((255,0,0))
        self.slider_fm_sizer.Add(self.fps_message_current, 0, wx.ALL, 5)
        
        self.slider_fm_panel.SetSizer(self.slider_fm_sizer)
        self.slider_fm_panel.Layout()
        
        # Creates the text box warning message panel (for both the animation/frame rate buttons) and its sizer
        self.warning_panel = wx.Panel(self,pos=(840+self.a,650),size=(600,100))
        self.warning_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Creates the warning message immediately below the animation panel
        self.warning = wx.StaticText(self.warning_panel,label = u' ')
        self.warning.SetForegroundColour((255,0,0))
        self.warning_sizer.Add(self.warning, 0, wx.ALL, 5)
        
        self.warning_panel.SetSizer(self.warning_sizer)
        self.warning_panel.Layout() 
        
        # Creates the text box warning message panel (for the tracer button) and its sizer
        self.tracer_warning_panel = wx.Panel(self,pos=(890+self.a,670),size=(600,100))
        self.tracer_warning_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Creates the tracer warning message below the animation panel
        self.tracer_warning = wx.StaticText(self.tracer_warning_panel,label = u' ')
        self.tracer_warning.SetForegroundColour((255,0,0))
        self.tracer_warning_sizer.Add(self.tracer_warning, 0, wx.ALL, 5)
        
        self.tracer_warning_panel.SetSizer(self.tracer_warning_sizer)
        self.tracer_warning_panel.Layout() 
        
        self.tracer_warning.SetLabel('If you wish to run the animation, please disable the tracers.')
        
        # Creates the slider move panel and its sizer
        self.slider_m_panel = wx.Panel(self,pos=(800+self.a,560),size=(600,100))
        self.slider_m_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Reverse button creation
        self.reverse_btn = wx.Button(self.slider_m_panel, label='Reverse')
        self.reverse_btn.Bind(wx.EVT_BUTTON, self.on_reverse)
        self.slider_m_sizer.Add(self.reverse_btn, 0, wx.ALL, 5)

        # Reverse step button creation
        self.reverse_step_btn = wx.Button(self.slider_m_panel, label='Step Backward')
        self.reverse_step_btn.Bind(wx.EVT_BUTTON, self.reverse_step)
        self.slider_m_sizer.Add(self.reverse_step_btn, 0, wx.ALL, 5)  
        
        # Pause button creation
        self.pause_btn = wx.Button(self.slider_m_panel, label='Pause')
        self.pause_btn.Bind(wx.EVT_BUTTON, self.on_pause)
        self.slider_m_sizer.Add(self.pause_btn, 0, wx.ALL, 5) 
        
        self.pause_btn.Disable()

        # Forward step button creation
        self.forward_step_btn = wx.Button(self.slider_m_panel, label='Step Forward')
        self.forward_step_btn.Bind(wx.EVT_BUTTON, self.forward_step)
        self.slider_m_sizer.Add(self.forward_step_btn, 0, wx.ALL, 5) 
        
        # Forward button creation
        self.forward_btn = wx.Button(self.slider_m_panel, label='Forward')
        self.forward_btn.Bind(wx.EVT_BUTTON, self.on_forward)
        self.slider_m_sizer.Add(self.forward_btn, 0, wx.ALL, 5) 
        
        self.slider_m_panel.SetSizer(self.slider_m_sizer)
        self.slider_m_panel.Layout()

        self.forward_btn.Disable()
        self.reverse_btn.Disable()
        

        # Creates the slider skip panel and its sizer
        self.slider_s_panel = wx.Panel(self,pos=(910+self.a,590),size=(1000,100))
        self.slider_s_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # First Point Button creation
        self.firstpoint_btn = wx.Button(self.slider_s_panel, label='First Point')
        self.firstpoint_btn.Bind(wx.EVT_BUTTON, self.on_first_point)
        self.slider_s_sizer.Add(self.firstpoint_btn, 0, wx.ALL, 5)        
        
        # TS Point Button creation
        self.tspoint_btn = wx.Button(self.slider_s_panel, label='Transition State')
        self.tspoint_btn.Bind(wx.EVT_BUTTON, self.on_ts)
        self.slider_s_sizer.Add(self.tspoint_btn, 0, wx.ALL, 5) 
        
        # Last Point Button creation
        self.lastpoint_btn = wx.Button(self.slider_s_panel, label='Last Point')
        self.lastpoint_btn.Bind(wx.EVT_BUTTON, self.on_last_point)
        self.slider_s_sizer.Add(self.lastpoint_btn, 0, wx.ALL, 5) 
        
        self.slider_s_panel.SetSizer(self.slider_s_sizer)
        self.slider_s_panel.Layout()
        
        # Creates the additional options panel and its sizer
        self.slider_x_panel = wx.Panel(self,pos=(850+self.a,620),size=(1000,100))
        self.slider_x_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # xyz file download button creation
        self.xyz_btn = wx.Button(self.slider_x_panel, label='Download XYZ File')
        self.xyz_btn.Bind(wx.EVT_BUTTON, self.on_xyz_request)
        self.slider_x_sizer.Add(self.xyz_btn, 0, wx.ALL, 5) 
        
        # ASE GUI button creation
        self.ase_gui_btn = wx.Button(self.slider_x_panel, label='View with ASE GUI')
        self.ase_gui_btn.Bind(wx.EVT_BUTTON, self.open_ase_gui)
        self.slider_x_sizer.Add(self.ase_gui_btn, 0, wx.ALL, 5) 
        
        # Tracer disabler/enabler button creation
        self.button_message = "Disable Tracers"
        self.tr_btn = wx.Button(self.slider_x_panel, label=self.button_message)
        self.tr_btn.Bind(wx.EVT_BUTTON, self.on_tr_btn)
        self.slider_x_sizer.Add(self.tr_btn, 0, wx.ALL, 5)
        
        self.slider_x_panel.SetSizer(self.slider_x_sizer)
        self.slider_x_panel.Layout()
        
        self.Show()

    ### Event handlers for all user interactions with the GUI
        
    # Handler for when the user moves the slider with their mouse    
    def on_slide(self,event):
        wx.CallAfter(self.pause)
        obtained_value = float("{:.{}f}".format(float(self.my_slider.GetValue()),4))
        if self.at_min == False and self.at_max == False:
            for line_value in self.cui_set.keys():
                s = float("{:.{}f}".format(self.cui_set.get(line_value),4))
                distance = abs(s-obtained_value)
                if distance < 0.015:
                    self.text_ctrl.SetValue(str(s))
                    obtained_value = float("{:.{}f}".format(float(self.my_slider.GetValue()),4))
                    start_distance = abs(obtained_value - self.slider_min)
                    end_distance = abs(obtained_value - self.slider_max)
                    if start_distance < 0.015:
                        wx.CallAfter(self.jump_to_start)
                    elif end_distance < 0.015:
                        wx.CallAfter(self.jump_to_end)
                    break
        elif self.at_min == True:
            self.text_ctrl.SetValue(str(list(self.cui_set.values())[2]))
            self.my_slider.SetValue(float(self.text_ctrl.GetValue()))
            self.at_min = False
        elif self.at_max == True:
            self.text_ctrl.SetValue(str(list(self.cui_set.values())[len(list(self.cui_set.values()))-3]))
            self.my_slider.SetValue(float(self.text_ctrl.GetValue())) 
            self.at_max = False
        if self.text_ctrl.GetValue() != '':
            self.my_slider.SetValue(float(self.text_ctrl.GetValue()))
        wx.CallAfter(self.structure_update)
    
    # Handler for when the user presses the "Jump to s value" button
    def on_jump(self,event):
        wx.CallAfter(self.pause)
        input_num = float("{:.{}f}".format(float(self.text_ctrl.GetValue()),4))
        if input_num <= self.slider_max and input_num >= self.slider_min:
            for line_value in self.cui_set.keys():
                s = self.cui_set.get(line_value)
                distance = abs(float(s)-input_num)
                if distance < 0.02:
                    set_value = float("{:.{}f}".format(float(s),4))
                    break
            self.my_slider.SetValue(set_value)
            wx.CallAfter(self.slider_update)
            self.text_ctrl.SetValue(str(set_value))
            wx.CallAfter(self.s_value_update)
            wx.CallAfter(self.structure_update)

    # Function for refreshing and updating the text box after a change in the value of s                
    def s_value_update(self):
        self.text_ctrl.Refresh()
        self.text_ctrl.Update()
    
    # Function for refreshing and updating the slider after a change in the value of s
    def slider_update(self):
        self.my_slider.Refresh()
        self.my_slider.Update()
    
    # Handler for when the user presses the "Disable Tracers" button (which becomes the "Enable Tracers" button after the tracers have been disabled)
    def on_tr_btn(self,event):
        if self.button_message == "Disable Tracers":
            self.tracers_enabled = False
            self.curv_plot.clear()
            self.curv_plot.plot(self.curv_coord,self.curv_val,bgcolor = "WHITE",framecolor="WHITE",xlabel="Reaction parameter " + u's [amu¹\u141F² Bohr]',ylabel="Curvature",ymin=0,ymax=20)
            self.energy_plot.clear()
            self.energy_plot.plot(self.energy_coord,self.energy_val,bgcolor = "WHITE",framecolor="WHITE",xlabel="Reaction parameter " + u's [amu¹\u141F² Bohr]',ylabel="Energy (kcal/mol)",ymin=(min(self.energy_val)-5),ymax=max(self.energy_val)+5)
            self.button_message = "Enable Tracers"
            self.tr_btn.SetLabel(self.button_message)
            self.tracers_present = False
            self.fr_text_ctrl.Enable()
            self.forward_btn.Enable()
            self.reverse_btn.Enable()
            self.pause_btn.Enable()
            self.tracer_warning.SetLabel(' ')
        elif self.button_message == "Enable Tracers":
            self.tracers_enabled = True
            wx.CallAfter(self.structure_update)
            self.button_message = "Disable Tracers"
            self.tr_btn.SetLabel(self.button_message)
            self.frame_rate = 25
            self.fr_text_ctrl.SetValue(str(self.frame_rate))
            self.fps_message_current.SetLabel('Current frame rate: ' + str(self.frame_rate) + ' fps')
            self.fr_text_ctrl.Disable()
            self.forward_btn.Disable()
            self.reverse_btn.Disable()
            self.pause_btn.Disable()
            self.tracer_warning.SetLabel('If you wish to run the animation, please disable the tracers.')
            
    # Function for refreshing and updating the 2D RC visual after a change in the value of s        
    def structure_update(self):
        s_val = float("{:.{}f}".format(float(self.my_slider.GetValue()),4))
        for line_value in self.cui_set.keys():
            s = self.cui_set.get(line_value)
            distance = abs(float(s)-s_val)
            if distance < 0.0015:
                target = s
                break
        png_name = "./frames/"+str(target)+".png"
        structure_image = wx.Image(png_name,wx.BITMAP_TYPE_ANY)
        aspect_ratio = structure_image.GetWidth()/structure_image.GetHeight()
        NewH = 400
        NewW = int(NewH * aspect_ratio) 
        structure_image = structure_image.Scale(NewW,NewH)
        self.img.SetBitmap(wx.Bitmap(structure_image))
        self.visual_panel.Refresh()
        self.visual_panel.Update()
        
        # The red tracer for the curvature and energy graphs will be updated as well
        if self.tracers_enabled == True:
            # Update curvature tracer
            self.curv_tracer_x = np.array([target,target])
            if self.tracers_present == False:
                self.curv_plot.oplot(self.curv_tracer_x,self.curv_tracer_y)
            self.curv_plot.update_line(1,self.curv_tracer_x,self.curv_tracer_y,update_limits=False,draw=True)
            self.curv_plot_panel.Refresh()
            self.curv_plot_panel.Update()
            # Update energy tracer
            self.energy_tracer_x = np.array([target,target])
            if self.tracers_present == False:
                self.energy_plot.oplot(self.energy_tracer_x,self.energy_tracer_y)
            self.energy_plot.update_line(1,self.energy_tracer_x,self.energy_tracer_y,update_limits=False,draw=True)
            self.energy_plot_panel.Refresh()
            self.energy_plot_panel.Update()
            self.tracers_present = True
    
    # Handler for when the user presses the "Step Backward" button 
    def reverse_step(self,event):
        wx.CallAfter(self.pause)          
        set_value = float("{:.{}f}".format(float(self.text_ctrl.GetValue()),4))
        cui_coords = list(self.cui_set.values())
        start_distance = abs(set_value - self.slider_min)
        if start_distance < 0.015 and self.at_min == False:
            wx.CallAfter(self.jump_to_start)
        elif self.at_min == False:
            for cui_coord in cui_coords:
                distance = abs(cui_coord-set_value)
                if distance < 0.015:
                    target = cui_coord
                    break
            previous_frame = cui_coords.index(target) - 1
            previous_value = float("{:.{}f}".format(cui_coords[previous_frame],4))
            current_value = float("{:.{}f}".format(cui_coords[previous_frame+1],4))
            if previous_frame != -1 and previous_value >= self.slider_min and previous_value != current_value:
                self.text_ctrl.SetValue(str(previous_value))
                self.my_slider.SetValue(float(self.text_ctrl.GetValue()))
                wx.CallAfter(self.structure_update)
            elif previous_value == current_value:
                for i in range(previous_frame-1, -1, -1):
                    second_previous_value = float("{:.{}f}".format(cui_coords[i],4))
                    previous_value = float("{:.{}f}".format(cui_coords[i+1],4))
                    if second_previous_value != previous_value:
                        self.text_ctrl.SetValue(str(second_previous_value))
                        self.my_slider.SetValue(float(self.text_ctrl.GetValue()))
                        wx.CallAfter(self.structure_update)
                        break
        self.at_max = False
        
    # Function for iteratively updating the text box, slider, and RC visual during the reverse animation of the reaction 
    def reverse(self):
        self.text_ctrl.Disable()
        self.fr_text_ctrl.Disable()  
        self.tr_alive = True
        set_value = float("{:.{}f}".format(float(self.text_ctrl.GetValue()),4))
        cui_coords = list(self.cui_set.values())
        for cui_coord in cui_coords:
            distance = abs(cui_coord-set_value)
            if distance < 0.015:
                target = cui_coord
                break
        previous_frame = cui_coords.index(target) - 1
        previous_value = float("{:.{}f}".format(cui_coords[previous_frame],4))
        current_value = float("{:.{}f}".format(cui_coords[previous_frame+1],4))
        while previous_frame >= 0:
            if previous_value >= self.slider_min and previous_value != current_value:
                time.sleep(1/self.frame_rate)
                self.text_ctrl.SetValue(str(previous_value))
                wx.CallAfter(self.s_value_update)
                self.my_slider.SetValue(float(self.text_ctrl.GetValue()))
                wx.CallAfter(self.slider_update)            
                previous_frame -= 1
                previous_value = float("{:.{}f}".format(cui_coords[previous_frame],4))
                current_value = float("{:.{}f}".format(cui_coords[previous_frame+1],4))                
                wx.CallAfter(self.structure_update)
                if previous_frame == self.first_frame:
                    self.text_ctrl.Enable()
                    self.fr_text_ctrl.Enable()
            elif previous_value == current_value:
                for i in range(previous_frame-1, -1, -1):
                    second_previous_value = float("{:.{}f}".format(cui_coords[i],4))
                    previous_value = float("{:.{}f}".format(cui_coords[i+1],4))
                    if second_previous_value != previous_value:
                        time.sleep(1/self.frame_rate)
                        self.text_ctrl.SetValue(str(second_previous_value))
                        wx.CallAfter(self.s_value_update)
                        self.my_slider.SetValue(float(self.text_ctrl.GetValue()))
                        wx.CallAfter(self.slider_update)            
                        previous_frame -= 1
                        previous_value = float("{:.{}f}".format(cui_coords[previous_frame],4))
                        current_value = float("{:.{}f}".format(cui_coords[previous_frame+1],4))
                        wx.CallAfter(self.structure_update)
                        if previous_frame == self.first_frame:
                            self.text_ctrl.Enable()
                            self.fr_text_ctrl.Enable()
                        break
        wx.CallAfter(self.jump_to_start)

    # Handler for when the user presses the "Reverse" button 
    def on_reverse(self,event):
        if self.text_ctrl.HasFocus():
            self.warning.SetLabel('Please deselect the s value text box before clicking the Reverse button.')
        if self.fr_text_ctrl.HasFocus():
            self.warning.SetLabel('Please deselect the frame rate text box before clicking the Reverse button.')
        if self.text_ctrl.HasFocus() == False and self.fr_text_ctrl.HasFocus() == False:    
            if (threading.active_count()-self.standby_thread_count)==0:
                global tr
                tr = thread_with_trace(target=self.reverse)
                tr.start()
                self.warning.SetLabel('')
            elif self.tf_alive:
                tf.kill()
                tf.join()
                self.tf_alive = False
                tr = thread_with_trace(target=self.reverse)
                tr.start()
                self.warning.SetLabel('')
            
    # Handler for when the user presses the "Step Forward" button        
    def forward_step(self,event):
        wx.CallAfter(self.pause)          
        set_value = float("{:.{}f}".format(float(self.text_ctrl.GetValue()),4))
        cui_coords = list(self.cui_set.values())
        end_distance = abs(set_value - self.slider_max)
        if end_distance < 0.015 and self.at_max == False:
            wx.CallAfter(self.jump_to_end)
        elif self.at_max == False:
            for cui_coord in cui_coords:
                distance = cui_coord-set_value
                if distance < 0.05 and distance > 0:
                    target = cui_coord
                    break
            next_frame = cui_coords.index(target) + 1
            next_value = float("{:.{}f}".format(cui_coords[next_frame],4))
            current_value = float("{:.{}f}".format(cui_coords[next_frame-1],4))
            if next_frame != len(cui_coords) and next_value <= self.slider_max and current_value != next_value:
                self.text_ctrl.SetValue(str(next_value))
                self.my_slider.SetValue(float(self.text_ctrl.GetValue()))
                wx.CallAfter(self.structure_update)
            elif current_value == next_value:
                for i in range(next_frame+1, len(cui_coords)):
                    second_next_value = float("{:.{}f}".format(cui_coords[i],4))
                    next_value = float("{:.{}f}".format(cui_coords[i-1],4))
                    if next_value != second_next_value:
                        self.text_ctrl.SetValue(str(second_next_value))
                        self.my_slider.SetValue(float(self.text_ctrl.GetValue()))
                        wx.CallAfter(self.structure_update)
                        break
        self.at_min = False
                    
    # Function for iteratively updating the text box, slider, and RC visual during the forward animation of the reaction  
    def forward(self):
        self.text_ctrl.Disable()
        self.fr_text_ctrl.Disable()        
        self.tf_alive = True
        set_value = float("{:.{}f}".format(float(self.text_ctrl.GetValue()),4))
        cui_coords = list(self.cui_set.values())
        for cui_coord in cui_coords:
            distance = abs(cui_coord-set_value)
            if distance < 0.015:
                target = cui_coord
                break
        next_frame = cui_coords.index(target) + 1
        next_value = float("{:.{}f}".format(cui_coords[next_frame],4))
        current_value = float("{:.{}f}".format(cui_coords[next_frame-1],4))
        while next_frame != len(cui_coords)-1:
            if next_value <= self.slider_max and current_value != next_value:
                time.sleep(1/self.frame_rate)
                self.text_ctrl.SetValue(str(next_value))
                wx.CallAfter(self.s_value_update)
                self.my_slider.SetValue(float(self.text_ctrl.GetValue()))
                wx.CallAfter(self.slider_update)            
                next_frame += 1
                next_value = float("{:.{}f}".format(cui_coords[next_frame],4))
                current_value = float("{:.{}f}".format(cui_coords[next_frame-1],4))
                wx.CallAfter(self.structure_update)
                if next_frame == self.last_frame:
                    self.text_ctrl.Enable()
                    self.fr_text_ctrl.Enable()
            elif next_value == current_value:
                for i in range(next_frame+1, len(cui_coords)):
                    second_next_value = float("{:.{}f}".format(cui_coords[i],4))
                    next_value = float("{:.{}f}".format(cui_coords[i-1],4))
                    if second_next_value != next_value:
                        time.sleep(1/self.frame_rate)
                        self.text_ctrl.SetValue(str(second_next_value))
                        wx.CallAfter(self.s_value_update)
                        self.my_slider.SetValue(float(self.text_ctrl.GetValue()))
                        wx.CallAfter(self.slider_update)            
                        next_frame += 1
                        next_value = float("{:.{}f}".format(cui_coords[next_frame],4))
                        current_value = float("{:.{}f}".format(cui_coords[next_frame-1],4))
                        wx.CallAfter(self.structure_update)
                        if next_frame == self.last_frame:
                            self.text_ctrl.Enable()
                            self.fr_text_ctrl.Enable()
                        break
        wx.CallAfter(self.jump_to_end)
    
    # Handler for when the user presses the "Forward" button
    def on_forward(self,event):
        if self.text_ctrl.HasFocus():
            self.warning.SetLabel('Please deselect the s value text box before clicking the Forward button.')
        if self.fr_text_ctrl.HasFocus():
            self.warning.SetLabel('Please deselect the frame rate text box before clicking the Forward button.')
        if self.text_ctrl.HasFocus() == False and self.fr_text_ctrl.HasFocus() == False:     
            if (threading.active_count()-self.standby_thread_count)==0:
                global tf
                tf = thread_with_trace(target=self.forward)
                tf.start()
                self.warning.SetLabel('')
            elif self.tr_alive:
                tr.kill()
                tr.join()
                self.tr_alive = False
                tf = thread_with_trace(target=self.forward)
                tf.start()
                self.warning.SetLabel('')
        
    # Function for pausing the forward or reverse animation of the reaction    
    def pause(self):
        if self.tf_alive:
                tf.kill()
                tf.join()
                self.tf_alive = False
                self.text_ctrl.Enable()
                if self.tracers_present == False:
                    self.fr_text_ctrl.Enable()
        if self.tr_alive:
                tr.kill()
                tr.join()
                self.tr_alive = False
                self.text_ctrl.Enable()
                if self.tracers_present == False:
                    self.fr_text_ctrl.Enable()

    # Handler for when the user presses the "Pause" button    
    def on_pause(self,event):
        wx.CallAfter(self.pause)
            
    # Handler for when the user presses the "Set Frame Rate" button        
    def on_fr_btn(self, event):
        self.frame_rate = math.floor(float(self.fr_text_ctrl.GetValue()))
        self.fr_text_ctrl.SetValue(str(self.frame_rate))
        self.fps_message_current.SetLabel('Current frame rate: ' + str(self.frame_rate) + ' fps')
    
    # Function for updating the text box, slider, and RC visual to the first s value along the reaction coordinate
    def jump_to_start(self):
        wx.CallAfter(self.pause)        
        set_value = "{:.{}f}".format(self.slider_min,4)
        self.my_slider.SetValue(float(set_value))
        wx.CallAfter(self.slider_update)
        self.text_ctrl.SetValue(set_value)
        wx.CallAfter(self.s_value_update)
        wx.CallAfter(self.structure_update)   
        self.at_min = True
    
    # Handler for when the user presses the "First Point" button 
    def on_first_point(self,event):
        wx.CallAfter(self.jump_to_start)        

    # Handler for when the user presses the "Transition State" button 
    def on_ts(self,event):
        wx.CallAfter(self.pause)
        self.at_max = False
        self.at_min = False
        set_value = "{:.{}f}".format(0,4)
        self.my_slider.SetValue(float(set_value))
        wx.CallAfter(self.slider_update)
        self.text_ctrl.SetValue(set_value)
        wx.CallAfter(self.s_value_update)
        wx.CallAfter(self.structure_update)
        
    # Function for updating the text box, slider, and RC visual to the last s value along the reaction coordinate    
    def jump_to_end(self):
        wx.CallAfter(self.pause)
        set_value = "{:.{}f}".format(self.slider_max,4)
        self.my_slider.SetValue(float(set_value))
        wx.CallAfter(self.slider_update)
        self.text_ctrl.SetValue(set_value)
        wx.CallAfter(self.s_value_update)
        wx.CallAfter(self.structure_update)
        self.at_max = True
        
    # Handler for when the user presses the "Last Point" button     
    def on_last_point(self,event):
        wx.CallAfter(self.jump_to_end)
           
    # Function for generating XYZ file for current value of s
    def generate_xyz(self):
        wx.CallAfter(self.pause)
        set_value = float("{:.{}f}".format(float(self.text_ctrl.GetValue()),4))
        for cui_coord_line in self.cui_set.keys():
            distance = abs(self.cui_set[cui_coord_line]-set_value)
            if distance < 0.0015:
                xyz_name = "s_"+str(set_value)+".xyz"       
                xyz_file = open(xyz_name, "w")
                for r in range(cui_coord_line-1,cui_coord_line + self.n_atoms + 1):
                    self.xyz_lines[r]
                    xyz_file.write(self.xyz_lines[r])
                xyz_file.close()
                break

    # Handler for when the user presses the "Download XYZ File" button           
    def on_xyz_request(self,event):
        wx.CallAfter(self.generate_xyz)
        
    # Handler for when the user presses the "View with ASE GUI" button    
    def open_ase_gui(self,event):
        wx.CallAfter(self.pause)
        set_value = float("{:.{}f}".format(float(self.text_ctrl.GetValue()),4))
        for cui_coord_line in self.cui_set.keys():
            distance = abs(self.cui_set[cui_coord_line]-set_value)
            if distance < 0.0015:
                xyz_name = "s_"+str(set_value)+".xyz"       
                xyz_file = open(xyz_name, "w")
                for r in range(cui_coord_line-1,cui_coord_line + self.n_atoms + 1):
                    self.xyz_lines[r]
                    xyz_file.write(self.xyz_lines[r])
                xyz_file.close()
                break
        atoms_structure = ase.io.read(xyz_name)
        view(atoms_structure)
        os.remove(xyz_name)
                
    # Handler for when the user presses return on their keyboard after entering in a new value of s in the text box below the slider            
    def onButtonKeyEvent(self, event):
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_RETURN:
            input_num = float("{:.{}f}".format(float(self.text_ctrl.GetValue()),4))
            if input_num <= self.slider_max and input_num >= self.slider_min:
                for line_value in self.cui_set.keys():
                    s = self.cui_set.get(line_value)
                    distance = abs(float(s)-input_num)
                    if distance < 0.02:
                        set_value = float("{:.{}f}".format(float(s),4))
                        if abs(set_value - float("{:.{}f}".format(float(self.slider_min),4))) <= 0.00001:
                            wx.CallAfter(self.jump_to_start)
                        if abs(set_value - float("{:.{}f}".format(float(self.slider_max),4))) <= 0.00001:
                            wx.CallAfter(self.jump_to_end)   
                        break
                self.my_slider.SetValue(set_value)
                wx.CallAfter(self.slider_update)
                self.text_ctrl.SetValue(str(set_value))
                wx.CallAfter(self.s_value_update)
                wx.CallAfter(self.structure_update)
        event.Skip()
    
    # Handler for when the user presses return on their keyboard after entering in a new frame rate in the frame rate text box  
    def onButtonKeyEvent_fr(self, event):
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_RETURN:
            self.frame_rate = math.floor(float(self.fr_text_ctrl.GetValue()))
            self.fr_text_ctrl.SetValue(str(self.frame_rate))
            self.fps_message_current.SetLabel('Current frame rate: ' + str(self.frame_rate) + ' fps')
        event.Skip()


# Program execution command
if __name__ == '__main__':

    sys.stdout = io.StringIO()
    
    app = wx.App()
    frame = MyFrame()
    app.MainLoop()

    sys.stdout = sys.__stdout__

