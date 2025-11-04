import os
import sys
import pandas as pd
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import subprocess

class CameraCalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Calibration Tool")
        self.root.geometry("1600x900")
        
        # Data storage
        self.satellite_image = None
        self.camera_image = None
        self.satellite_points = []  # (x, y) coordinates
        self.camera_points = []     # (x, y) coordinates
        self.origin_point = None    # (lat, lon) tuple
        self.geo_points = []        # (lat, lon) for each point
        
        # UI setup
        self._create_menu()
        self._create_main_layout()
        
        # Status variables
        self.max_points = 6
        self.current_mode = "satellite"  # "satellite" or "camera"
        
    def _create_menu(self):
        menu_bar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Load Satellite Image", command=self._load_satellite_image)
        file_menu.add_command(label="Load Camera Image", command=self._load_camera_image)
        file_menu.add_separator()
        file_menu.add_command(label="Save Calibration Data", command=self._save_calibration_data)
        file_menu.add_command(label="Load Calibration Data", command=self._load_calibration_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Calibration menu
        calib_menu = tk.Menu(menu_bar, tearoff=0)
        calib_menu.add_command(label="Set Origin Point", command=self._set_origin_mode)
        calib_menu.add_command(label="Clear All Points", command=self._clear_all_points)
        calib_menu.add_command(label="Run Calibration", command=self._run_calibration)
        
        # Add menus to the menu bar
        menu_bar.add_cascade(label="File", menu=file_menu)
        menu_bar.add_cascade(label="Calibration", menu=calib_menu)
        
        self.root.config(menu=menu_bar)
    
    def _create_main_layout(self):
        # Main frame to contain everything
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Split the main frame into left and right sections
        left_frame = tk.Frame(main_frame, width=750)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        right_frame = tk.Frame(main_frame, width=750)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Satellite view (left)
        self.satellite_fig, self.satellite_ax = plt.subplots(figsize=(7, 7))
        self.satellite_canvas = FigureCanvasTkAgg(self.satellite_fig, left_frame)
        self.satellite_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.satellite_ax.set_title("Satellite View (Map)")
        self.satellite_ax.set_xticks([])
        self.satellite_ax.set_yticks([])
        
        # Button to activate satellite point selection
        self.satellite_btn = tk.Button(
            left_frame, 
            text="Select Points on Satellite Image", 
            command=self._activate_satellite_mode
        )
        self.satellite_btn.pack(pady=5)
        
        # Camera view (right)
        self.camera_fig, self.camera_ax = plt.subplots(figsize=(7, 7))
        self.camera_canvas = FigureCanvasTkAgg(self.camera_fig, right_frame)
        self.camera_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.camera_ax.set_title("Camera View")
        self.camera_ax.set_xticks([])
        self.camera_ax.set_yticks([])
        
        # Button to activate camera point selection
        self.camera_btn = tk.Button(
            right_frame, 
            text="Select Points on Camera Image", 
            command=self._activate_camera_mode
        )
        self.camera_btn.pack(pady=5)
        
        # Status frame at bottom
        status_frame = tk.Frame(self.root)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = tk.Label(
            status_frame, 
            text="Load satellite and camera images to begin calibration", 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X)
        
        # Set up click events and zoom functionality
        self.satellite_canvas.mpl_connect('button_press_event', self._on_satellite_click)
        self.camera_canvas.mpl_connect('button_press_event', self._on_camera_click)
        
        # Add zoom and navigation functionality
        self.satellite_toolbar = self._add_toolbar(self.satellite_canvas, left_frame)
        self.camera_toolbar = self._add_toolbar(self.camera_canvas, right_frame)
        
    def _add_toolbar(self, canvas, parent):
        toolbar_frame = tk.Frame(parent)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        # Add keyboard shortcuts and mouse wheel for zoom
        canvas.get_tk_widget().bind("<Control-equal>", lambda event: self._zoom(canvas, 1.2))
        canvas.get_tk_widget().bind("<Control-minus>", lambda event: self._zoom(canvas, 0.8))
        canvas.get_tk_widget().bind("<Control-0>", lambda event: self._reset_zoom(canvas))
        canvas.get_tk_widget().bind("<Control-equal>", lambda _: self._zoom(canvas, 1.2))
        canvas.get_tk_widget().bind("<Control-minus>", lambda _: self._zoom(canvas, 0.8))
        canvas.get_tk_widget().bind("<Control-0>", lambda _: self._reset_zoom(canvas))
            if event.state & 0x4:  # Check if Ctrl key is pressed (0x4 is the state flag for Ctrl)
                if event.delta > 0:
                    self._zoom(canvas, 1.1)  # Zoom in
                else:
                    self._zoom(canvas, 0.9)  # Zoom out
        
        canvas.get_tk_widget().bind("<MouseWheel>", _on_mousewheel)  # Windows and macOS
        canvas.get_tk_widget().bind("<Button-4>", lambda event: _on_mousewheel(type('event', (), {'delta': 120, 'state': 0x4})))  # Linux scroll up
        canvas.get_tk_widget().bind("<Button-5>", lambda event: _on_mousewheel(type('event', (), {'delta': -120, 'state': 0x4})))  # Linux scroll down
        
        canvas.get_tk_widget().bind("<Button-4>", lambda _: _on_mousewheel(type('event', (), {'delta': 120, 'state': 0x4})))  # Linux scroll up
        canvas.get_tk_widget().bind("<Button-5>", lambda _: _on_mousewheel(type('event', (), {'delta': -120, 'state': 0x4})))  # Linux scroll down
    def _zoom(self, canvas, factor):
        """Zoom in/out of the plot."""
        ax = canvas.figure.axes[0]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        new_width = (xlim[1] - xlim[0]) / factor
        new_height = (ylim[1] - ylim[0]) / factor
        
        xmid = (xlim[1] + xlim[0]) / 2
        ymid = (ylim[1] + ylim[0]) / 2
        
        ax.set_xlim(xmid - new_width/2, xmid + new_width/2)
        ax.set_ylim(ymid - new_height/2, ymid + new_height/2)
        canvas.draw_idle()
    
    def _reset_zoom(self, canvas):
        """Reset zoom to show the whole image."""
        ax = canvas.figure.axes[0]
        ax.autoscale(True)
        canvas.draw_idle()
    
    def _load_satellite_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Satellite Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_path:
            return
        
        try:
            self.satellite_image = cv2.imread(file_path)
            self.satellite_image = cv2.cvtColor(self.satellite_image, cv2.COLOR_BGR2RGB)
            
            # Update plot
            self.satellite_ax.clear()
            self.satellite_ax.imshow(self.satellite_image)
            self.satellite_ax.set_title("Satellite View (Map)")
            self.satellite_ax.set_xticks([])
            self.satellite_ax.set_yticks([])
            self.satellite_canvas.draw()
            
            # Reset points
            self.satellite_points = []
            self.geo_points = []
            self.origin_point = None
            self._update_plot_points()
            
            self.status_label.config(text=f"Loaded satellite image: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def _load_camera_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Camera Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_path:
            return
        
        try:
            self.camera_image = cv2.imread(file_path)
            self.camera_image = cv2.cvtColor(self.camera_image, cv2.COLOR_BGR2RGB)
            
            # Update plot
            self.camera_ax.clear()
            self.camera_ax.imshow(self.camera_image)
            self.camera_ax.set_title("Camera View")
            self.camera_ax.set_xticks([])
            self.camera_ax.set_yticks([])
            self.camera_canvas.draw()
            
            # Reset points
            self.camera_points = []
            self._update_plot_points()
            
            self.status_label.config(text=f"Loaded camera image: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def _activate_satellite_mode(self):
        self.current_mode = "satellite"
        self.satellite_btn.config(relief=tk.SUNKEN)
        self.camera_btn.config(relief=tk.RAISED)
        self.status_label.config(text="Select points on satellite image. Use toolbar or Ctrl+mouse wheel to zoom.")
    
    def _activate_camera_mode(self):
        self.current_mode = "camera"
        self.satellite_btn.config(relief=tk.RAISED)
        self.camera_btn.config(relief=tk.SUNKEN)
        self.status_label.config(text="Select points on camera image. Use toolbar or Ctrl+mouse wheel to zoom.")
    
    def _set_origin_mode(self):
        self.current_mode = "origin"
        self.satellite_btn.config(relief=tk.RAISED)
        self.camera_btn.config(relief=tk.RAISED)
        self.status_label.config(text="Click on satellite image to set origin point")
    
    def _on_satellite_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        
        x, y = event.xdata, event.ydata
        
        if self.current_mode == "origin":
            # Ask for latitude and longitude
            dialog = GeoCoordsDialog(self.root)
            self.root.wait_window(dialog.top)
            
            if dialog.lat is not None and dialog.lon is not None:
                self.origin_point = (dialog.lat, dialog.lon)
                self.status_label.config(text=f"Origin set at lat: {dialog.lat}, lon: {dialog.lon}")
                
                # Mark origin on satellite image
                self.satellite_ax.plot(x, y, 'rx', markersize=10)
                self.satellite_canvas.draw()
        
        elif self.current_mode == "satellite":
            if len(self.satellite_points) >= self.max_points:
                messagebox.showinfo("Info", f"Maximum {self.max_points} points allowed. Clear points to start over.")
                return
            
            # Ask for latitude and longitude using a fixed dialog
            dialog = GeoCoordsDialog(self.root)
            self.root.wait_window(dialog.top)
            
            # Only proceed if valid coordinates were entered
            if dialog.lat is not None and dialog.lon is not None:
                self.satellite_points.append((x, y))
                self.geo_points.append((dialog.lat, dialog.lon))
                self.status_label.config(text=f"Added point {len(self.satellite_points)} at lat: {dialog.lat}, lon: {dialog.lon}")
                self._update_plot_points()
    
    def _on_camera_click(self, event):
        if self.current_mode != "camera" or self.camera_image is None:
            return
        
        if event.xdata is None or event.ydata is None:
            return
        
        x, y = event.xdata, event.ydata
        
        if len(self.camera_points) >= self.max_points:
            messagebox.showinfo("Info", f"Maximum {self.max_points} points allowed. Clear points to start over.")
            return
        
        self.camera_points.append((x, y))
        self.status_label.config(text=f"Added camera point {len(self.camera_points)} at x: {x:.1f}, y: {y:.1f}")
        self._update_plot_points()
    
    def _update_plot_points(self):
        # Save current zoom limits before clearing
        sat_xlim = self.satellite_ax.get_xlim() if self.satellite_image is not None else None
        sat_ylim = self.satellite_ax.get_ylim() if self.satellite_image is not None else None
        cam_xlim = self.camera_ax.get_xlim() if self.camera_image is not None else None
        cam_ylim = self.camera_ax.get_ylim() if self.camera_image is not None else None
        
        # Update satellite points
        self.satellite_ax.clear()
        if self.satellite_image is not None:
            self.satellite_ax.imshow(self.satellite_image)
            
            # Restore zoom level if it was set
            if sat_xlim and sat_ylim and self.satellite_points:
                self.satellite_ax.set_xlim(sat_xlim)
                self.satellite_ax.set_ylim(sat_ylim)
        
        # Draw satellite points
        for i, (x, y) in enumerate(self.satellite_points):
            self.satellite_ax.plot(x, y, 'ro')
            self.satellite_ax.text(x+5, y+5, str(i+1), color='white', fontsize=12, 
                                 bbox=dict(facecolor='red', alpha=0.7))
        
        # Draw origin point if exists
        if self.origin_point and len(self.satellite_points) > 0:  # Use first point to mark origin
            x, y = self.satellite_points[0]
            self.satellite_ax.plot(x, y, 'rx', markersize=10)
            self.satellite_ax.text(x+5, y-15, "Origin", color='white', fontsize=12,
                                 bbox=dict(facecolor='blue', alpha=0.7))
        
        self.satellite_ax.set_title("Satellite View (Map)")
        self.satellite_ax.set_xticks([])
        self.satellite_ax.set_yticks([])
        self.satellite_canvas.draw()
        
        # Update camera points
        self.camera_ax.clear()
        if self.camera_image is not None:
            self.camera_ax.imshow(self.camera_image)
            
            # Restore zoom level if it was set
            if cam_xlim and cam_ylim and self.camera_points:
                self.camera_ax.set_xlim(cam_xlim)
                self.camera_ax.set_ylim(cam_ylim)
        
        # Draw camera points
        for i, (x, y) in enumerate(self.camera_points):
            self.camera_ax.plot(x, y, 'go')
            self.camera_ax.text(x+5, y+5, str(i+1), color='white', fontsize=12, 
                               bbox=dict(facecolor='green', alpha=0.7))
        
        self.camera_ax.set_title("Camera View")
        self.camera_ax.set_xticks([])
        self.camera_ax.set_yticks([])
        self.camera_canvas.draw()
    
    def _clear_all_points(self):
        self.satellite_points = []
        self.camera_points = []
        self.geo_points = []
        self.origin_point = None
        self._update_plot_points()
        self.status_label.config(text="All points cleared")
    
    def _save_calibration_data(self):
        if not self.satellite_points or not self.camera_points or not self.geo_points:
            messagebox.showerror("Error", "No calibration data to save")
            return
        
        if len(self.satellite_points) != len(self.camera_points):
            messagebox.showerror("Error", "Number of satellite and camera points must be equal")
            return
        
        if not self.origin_point:
            messagebox.showerror("Error", "Origin point not set")
            return
        
        # Create dataframe
        data = {
            'x': [p[0] for p in self.camera_points],
            'y': [p[1] for p in self.camera_points],
            'lat': [p[0] for p in self.geo_points],
            'lon': [p[1] for p in self.geo_points],
            'origin_lat': [self.origin_point[0]] * len(self.geo_points),
            'origin_lon': [self.origin_point[1]] * len(self.geo_points)
        }
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        file_path = filedialog.asksaveasfilename(
            title="Save Calibration Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if not file_path:
            return
        
        try:
            df.to_csv(file_path, index=False)
            self.status_label.config(text=f"Saved calibration data to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")
    
    def _load_calibration_data(self):
        file_path = filedialog.askopenfilename(
            title="Load Calibration Data",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if not file_path:
            return
        
        try:
            df = pd.read_csv(file_path)
            
            # Check if all required columns exist
            required_cols = ['x', 'y', 'lat', 'lon', 'origin_lat', 'origin_lon']
            if not all(col in df.columns for col in required_cols):
                messagebox.showerror("Error", "Invalid CSV format")
                return
            
            # Load data
            self.camera_points = [(row['x'], row['y']) for _, row in df.iterrows()]
            self.geo_points = [(row['lat'], row['lon']) for _, row in df.iterrows()]
            
            # Use the first row's origin
            self.origin_point = (df['origin_lat'].iloc[0], df['origin_lon'].iloc[0])
            
            # For satellite points, we don't have the exact image coordinates, so we'll use placeholder
            self.satellite_points = [(0, 0)] * len(self.camera_points)
            
            self._update_plot_points()
            self.status_label.config(text=f"Loaded calibration data from {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def _run_calibration(self):
        if not self.camera_image:
            messagebox.showerror("Error", "No camera image loaded")
            return
        
        if not self.satellite_points or not self.camera_points or not self.geo_points:
            messagebox.showerror("Error", "No calibration data available")
            return
        
        if len(self.camera_points) < 4:
            messagebox.showerror("Error", "Need at least 4 points for calibration")
            return
        
        # Save current camera image
        temp_img_path = "temp_camera_image.jpg"
        cv2.imwrite(temp_img_path, cv2.cvtColor(self.camera_image, cv2.COLOR_RGB2BGR))
        
        # Save calibration data
        temp_csv_path = "temp_calibration_data.csv"
        data = {
            'x': [p[0] for p in self.camera_points],
            'y': [p[1] for p in self.camera_points],
            'lat': [p[0] for p in self.geo_points],
            'lon': [p[1] for p in self.geo_points],
            'origin_lat': [self.origin_point[0]] * len(self.geo_points),
            'origin_lon': [self.origin_point[1]] * len(self.geo_points)
        }
        pd.DataFrame(data).to_csv(temp_csv_path, index=False)
        
        # Run calibration script
        try:
            # calibration script located at ../calibrate_camera.py relative to this script
            calibration_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "calibrate_camera.py"))
            
            cmd = [
                sys.executable,
                calibration_script,
                "--image", temp_img_path,
                "--csv", temp_csv_path,
                "--method", "anycalib"
            ]
            self.status_label.config(text="Running calibration...")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Calibration failed: {stderr.decode()}")
            self.status_label.config(text="Calibration completed. Camera model saved as camera_model.yml")
            
        except Exception as e:
            messagebox.showerror("Error", f"Calibration failed: {str(e)}")
        finally:
            # Clean up temp files
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)


class GeoCoordsDialog:
    def __init__(self, parent):
        self.lat = None
        self.lon = None
        
        self.top = tk.Toplevel(parent)
        self.top.title("Enter Geographic Coordinates")
        self.top.geometry("300x150")
        self.top.resizable(False, False)
        self.top.transient(parent)  # Make it a transient window (always on top of parent)
        self.top.grab_set()  # Modal dialog
        
        # Center on parent
        x = parent.winfo_rootx() + parent.winfo_width() // 2 - 150
        y = parent.winfo_rooty() + parent.winfo_height() // 2 - 75
        self.top.geometry(f"+{x}+{y}")
        
        # Latitude
        lat_frame = tk.Frame(self.top)
        lat_frame.pack(fill=tk.X, padx=20, pady=(20, 5))
        
        lat_label = tk.Label(lat_frame, text="Latitude:")
        lat_label.pack(side=tk.LEFT)
        
        self.lat_entry = tk.Entry(lat_frame)
        self.lat_entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
        
        # Longitude
        lon_frame = tk.Frame(self.top)
        lon_frame.pack(fill=tk.X, padx=20, pady=5)
        
        lon_label = tk.Label(lon_frame, text="Longitude:")
        lon_label.pack(side=tk.LEFT)
        
        self.lon_entry = tk.Entry(lon_frame)
        self.lon_entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
        
        # Buttons
        btn_frame = tk.Frame(self.top)
        btn_frame.pack(fill=tk.X, padx=20, pady=(5, 20))
        
        ok_btn = tk.Button(btn_frame, text="OK", command=self._on_ok)
        ok_btn.pack(side=tk.RIGHT, padx=5)
        
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=self._on_cancel)
        cancel_btn.pack(side=tk.RIGHT, padx=5)
        
        # Set keyboard bindings
        self.top.bind("<Return>", lambda event: self._on_ok())
        self.top.bind("<Escape>", lambda event: self._on_cancel())
        self.top.bind("<Return>", lambda _: self._on_ok())
        self.top.bind("<Escape>", lambda _: self._on_cancel())
        self.lat_entry.focus_set()
    
    def _on_ok(self):
        try:
            # Make sure we have valid float values
            lat_val = self.lat_entry.get().strip()
            lon_val = self.lon_entry.get().strip()
            
            if not lat_val or not lon_val:
                messagebox.showerror("Error", "Please enter both latitude and longitude")
                return
                
            self.lat = float(lat_val)
            self.lon = float(lon_val)
            self.top.destroy()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric coordinates")
    
    def _on_cancel(self):
        self.top.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraCalibrationApp(root)
    root.mainloop()