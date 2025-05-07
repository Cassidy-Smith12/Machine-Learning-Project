import os
import traceback
import sys
import io
import contextlib
from PyQt6.QtWidgets import (QMainWindow, QScrollArea, QWidget, QVBoxLayout, QComboBox, 
                            QPushButton, QApplication, QFileDialog, QLabel,
                            QTextEdit, QDialog, QSizePolicy, QMessageBox, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

try:
    import algorithms
except ImportError:
    print("Error: algorithms.py not found. Make sure it's in the same directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error importing algorithms.py: {e}")
    traceback.print_exc()
    sys.exit(1)



def add_plot_to_layout(plot_path, title, layout_to_add_to, scale_width=600, scale_height=450):
    # ---- Adds a plot image to the specified layout if it exists ----
    plot_widget = QWidget() # Container for title + image
    plot_layout = QVBoxLayout(plot_widget)
    plot_layout.setContentsMargins(0, 5, 0, 5) # margins

    plot_title_label = QLabel(f"<b>{title}</b>")
    plot_layout.addWidget(plot_title_label)

    if plot_path and os.path.exists(plot_path):
        plot_label = QLabel()
        pixmap = QPixmap(plot_path)
        # Scale pixmap smoothly
        scaled_pixmap = pixmap.scaled(scale_width, scale_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        plot_label.setPixmap(scaled_pixmap)
        plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plot_layout.addWidget(plot_label)
        print(f"Successfully added plot '{title}' from {plot_path}")
    else:
        print(f"Error: Plot not found or path not provided for '{title}': {plot_path}")
        missing_label = QLabel(f"<i>'{title}' plot not generated or found at expected path.</i>")
        missing_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plot_layout.addWidget(missing_label)

    layout_to_add_to.addWidget(plot_widget) # Add the container widget


class ResultsWindow(QDialog):
    def __init__(self, output_text, parent=None, metrics_text=None,
                 selected_algorithm=None, algorithm_output_dir=None,
                 next_day_prediction=None, is_error=False):
        super().__init__(parent)
        self.setWindowTitle(f"{selected_algorithm} Results" if selected_algorithm else "Results")
        self.setMinimumSize(1000, 800)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        # Store passed info
        self.selected_algorithm = selected_algorithm
        self.algorithm_output_dir = algorithm_output_dir
        self.next_day_prediction = next_day_prediction # Store the prediction
        self.is_error = is_error


        # Main scrollable area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #ffffff; }") # White background
        scroll_area_widget = QWidget()
        scroll_area_widget.setStyleSheet("background-color: #ffffff;")
        self.scroll_layout = QVBoxLayout(scroll_area_widget)  # Layout for scrollable content
        self.scroll_layout.setSpacing(20)                     # Create spacing between sections
        self.scroll_layout.setContentsMargins(10, 10, 10, 10) # Margins inside scroll area
        scroll_area.setWidget(scroll_area_widget)
        layout.addWidget(scroll_area, 1)                      # Add scroll area to results window


        # --- Algorithm Console Output --- (either error if error flag is true or correct output)
        console_label_text = "<b>Error Details:</b>" if self.is_error else "<b>Algorithm Console Output:</b>"
        results_label = QLabel(console_label_text)
        self.scroll_layout.addWidget(results_label)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setText(output_text)
        # Flexible height
        self.results_text.setMinimumHeight(150)
        # self.results_text.setMaximumHeight(300) #Limit max initial height
        self.results_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        self.scroll_layout.addWidget(self.results_text)


        # --- Metrics Display (Only if not an error window and metrics exist) ---
        if not self.is_error and metrics_text:
            metrics_label_title = QLabel("<b>Evaluation Metrics:</b>")
            self.scroll_layout.addWidget(metrics_label_title)
            metrics_display = QTextEdit()
            metrics_display.setReadOnly(True)
            metrics_display.setText(metrics_text)
            metrics_display.setFixedHeight(130) # Fixed height for metrics
            metrics_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.scroll_layout.addWidget(metrics_display)
        elif not self.is_error:
             # Still add a label if metrics are missing in a normal run, but notify user
             missing_metrics = QLabel("<i>Metrics file (metrics.txt) not found or not generated.</i>")
             missing_metrics.setAlignment(Qt.AlignmentFlag.AlignCenter)
             self.scroll_layout.addWidget(missing_metrics)
        


        # --- Section: Next Day Prediction (Only if not error and prediction exists) ---
        if not self.is_error and self.next_day_prediction is not None:
             prediction_frame = QFrame() # Use a frame for visual separation
             prediction_frame.setFrameShape(QFrame.Shape.StyledPanel)
             prediction_frame.setFrameShadow(QFrame.Shadow.Raised)
             prediction_layout = QVBoxLayout(prediction_frame)

             prediction_title = QLabel("<b>Next Day Price Prediction:</b>")
             prediction_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
             prediction_layout.addWidget(prediction_title)

             # Format the prediction as currency
             prediction_text = f"${self.next_day_prediction:,.2f}"
       

             prediction_value_label = QLabel(prediction_text)
             # Make prediction stand out
             prediction_value_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #006400; /* DarkGreen */")
             prediction_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
             prediction_layout.addWidget(prediction_value_label)

             self.scroll_layout.addWidget(prediction_frame) # Add the frame to the main scroll layout

        elif not self.is_error:
             # Notify if prediction is missing
             missing_prediction = QLabel("<i>Next day prediction not available or not generated.</i>")
             missing_prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)
             self.scroll_layout.addWidget(missing_prediction)



        # --- Section: Generated Plot (Only if not error and output dir exists) ---
        if not self.is_error and self.algorithm_output_dir:
            plots_section_label = QLabel("<b>Generated Plots:</b>")
            plots_section_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            plots_section_label.setStyleSheet("font-size: 16px; margin-top: 10px; margin-bottom: 5px;")
            self.scroll_layout.addWidget(plots_section_label)

            print(f"Results window checking for plots in: {self.algorithm_output_dir}\n")
            # Define paths to saved image output
            actual_vs_predicted_path = os.path.join(self.algorithm_output_dir, "actual_vs_predicted.png")

            add_plot_to_layout(actual_vs_predicted_path, "Actual vs Predicted", self.scroll_layout)



                  # --- Close Button ---
        close_button = QPushButton("Close")
        close_button.setMinimumWidth(100) # Button Size
        close_button.setStyleSheet("background-color: red; color: white;") 
        close_button.clicked.connect(self.accept) 
        layout.addWidget(close_button, 0, Qt.AlignmentFlag.AlignRight) #Add to layout, aligned right


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset_path = None

        self.setWindowTitle("Bitcoin Price Prediction")
        self.setMinimumSize(550, 450) # Set minimum size
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0; /* Light grey background */
            }
            QWidget {
                /* Using Segoe UI if available, otherwise use Arial */
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px; 
                color: #333333; /* Dark grey text */
            }
            QLabel {
                margin-bottom: 4px; /* Space below labels */
                font-weight: bold; /* Make labels bold */
            }
             QLabel#DatasetStatusLabel, QLabel#PredictDatasetStatusLabel { /* For dataset label */
                font-weight: normal;
                margin-top: 5px;
                margin-bottom: 10px;
             }
            QComboBox {
                padding: 6px 10px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
                min-height: 25px; /* Ensure decent height */
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }

            /* So the drop down menu isn't gray on certain systems leaving options hard to read. */
            QComboBox QAbstractItemView {
                background-color: white;
                color: black;
                selection-background-color: #e0e0e0;  /* #007BFF */
                selection-color: black;
            }

            /* Can't figure out how to get arrow on drop down
                But we could import our own icon or image
            QComboBox::down-arrow {
              
            }
            */
            
            QPushButton {
                border-radius: 5px;
                background-color: #007BFF; /* Bold Blue */
                color: white;
                padding: 9px 18px; /* Increased padding */
                font-weight: bold;
                border: none;
                min-height: 25px; /* Ensure decent height */
                outline: none; /* Remove focus outline */
            }
            QPushButton:hover {
                background-color: #0958ad; /* Darker Green on hover */
            }
            QPushButton:pressed {
                background-color: #052f5c; /* Even darker Blue when pressed */
            }
            QPushButton:disabled {
                background-color: #cccccc; /* Grey when disabled */
                color: #666666;
            }
            QTextEdit { /* Style for text edits in ResultsWindow */
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #ffffff; /* White background */
                padding: 8px;
                font-family: Consolas, monospace; /* Monospace font for code/output */
            }
            QDialog { /* Style for the ResultsWindow dialog */
                 background-color: #f8f8f8; 
            }
            QFrame { /* Style for frames used as containers */
                 border: 1px solid #d0d0d0;
                 border-radius: 4px;
                 margin-top: 10px;
                 margin-bottom: 10px;
                 padding: 10px;
            }
            QScrollArea {
                 border: none;
            }
        """)


        # --- Main widget and layout ---
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)           # Apply layout to container
        layout.setContentsMargins(30, 30, 30, 30) # Generous margins
        layout.setSpacing(18)                     # Spacing between widget groups

         # --- 1. Algorithm selection ---
        layout.addWidget(QLabel("1. Select Algorithm:"))
        self.algorithm_combo = QComboBox()
        # Algorithm Options
        self.algorithm_display_names = [
            "Random Forest", "Support Vector Machine", "Linear Regression"
        ]

        self.algorithm_class_names = [
            "RandomForest", "SupportVectorMachine", "LinearRegression"
        ]

        self.algorithm_map = {}
        for display_name, class_name in zip(self.algorithm_display_names, self.algorithm_class_names):
            if hasattr(algorithms, class_name):
                self.algorithm_map[display_name] = getattr(algorithms, class_name)

        # add algorithm names to combo box
        self.algorithm_combo.addItems(self.algorithm_display_names)

        layout.addWidget(self.algorithm_combo)

        # --- 2. Dataset Upload ---
        layout.addWidget(QLabel("2. Upload Dataset:"))
        self.upload_button = QPushButton("Click to Upload CSV Dataset")
        self.upload_button.clicked.connect(self.upload_dataset)
        layout.addWidget(self.upload_button)

        # Dataset status label
        self.dataset_label = QLabel("No dataset selected.")
        self.dataset_label.setObjectName("DatasetStatusLabel") 
        self.dataset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dataset_label.setStyleSheet("font-style: italic; color: #555;") #Gray
        layout.addWidget(self.dataset_label)


        # --- 3. Run Analysis ---
        layout.addWidget(QLabel("3. Run Analysis & Prediction:"))
        self.run_button = QPushButton("Run Selected Algorithm")
        self.run_button.clicked.connect(self.run_algorithm)
        self.run_button.setEnabled(False)  # Disabled until dataset is uploaded
        layout.addWidget(self.run_button)

        layout.addStretch(1) # Add flexible space to push run button down

        

    def upload_dataset(self):
        # Use a more specific filter and allow All Files as fallback
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV Dataset",
            "", # Start directory is default
            "CSV Files (*.csv);;All Files (*)" # Filter for CSV files
        )
        if file_name:
            # Check extension case-insensitively
            if file_name.lower().endswith('.csv'):
                self.dataset_path = file_name
                # Display only the filename
                base_name = os.path.basename(file_name)
                self.dataset_label.setText(f"Dataset: {base_name}")
                # Use style sheet to show uploaded dataset
                self.dataset_label.setStyleSheet("color: #2E8B57; font-weight: bold; font-style: normal;") # SeaGreen, bold
                self.run_button.setEnabled(True) # Enable run button
            else:
                self.dataset_path = None
                self.dataset_label.setText("Invalid file type. Please upload a CSV file.")
                self.dataset_label.setStyleSheet("color: #DC143C; font-weight: bold; font-style: normal;") # Crimson red, bold
                self.run_button.setEnabled(False)
                QMessageBox.warning(self, "Invalid File", "Please select a valid CSV file (.csv extension).")
        # else: keep previous state



    def run_algorithm(self):

         # --- Reset prediction state whenever a new training run starts ---
      
        selected_display_name = self.algorithm_combo.currentText()
        AlgorithmClass = self.algorithm_map.get(selected_display_name)

        # Disable buttons during run
        self.run_button.setEnabled(False)
        self.upload_button.setEnabled(False)
        self.algorithm_combo.setEnabled(False)
        QApplication.processEvents() # Update UI

        output_text = ""
        metrics_text = None
        algo_output_dir = None
        error_occurred = False
        algo_instance = None # Define algo_instance here
        next_day_prediction_result = None # Store future prediction


         # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                print(f"--- Initializing {selected_display_name} ---")
                # algorithms.py called to create algorithm constructor
                algo_instance = AlgorithmClass(self.dataset_path)
                algo_output_dir = algo_instance.get_output_dir() # Get output dir early

                # Train the model
                print(f"\n--- Training {selected_display_name} ---")
                algo_instance.train()

                # Evaluate the model
                print(f"\n--- Evaluating {selected_display_name} ---")
                algo_instance.evaluate()

                # Retrieve the future prediction result stored by evaluate()
                next_day_prediction_result = getattr(algo_instance, 'next_day_prediction', None)


                print("\n--- Run Finished ---")

            # Combine stdout and stderr for the results window
            output_text = stdout_capture.getvalue() + "\n--- Errors/Warnings (if any) ---\n" + stderr_capture.getvalue()

            # Try reading the metrics file generated by evaluate()
            metrics_file_path = os.path.join(algo_output_dir, "metrics.txt")
            if os.path.exists(metrics_file_path):
                with open(metrics_file_path, 'r') as f:
                    metrics_text = f.read()
            else:
                print(f"Warning: metrics.txt not found in {algo_output_dir}")
                metrics_text = "Metrics file not found."

            

        except Exception as e:
            error_occurred = True
            print(f"An error occurred during {selected_display_name} execution:")
            traceback.print_exc() # Print full traceback to console
            # Combine stdout/stderr captured so far with the error message
            error_details = f"--- Error during execution ---\n{traceback.format_exc()}"
            output_text = stdout_capture.getvalue() + "\n--- Errors/Warnings ---\n" + stderr_capture.getvalue() + "\n" + error_details
            # Ensure output dir is cleared or handled if error happened mid-way
            algo_output_dir = None # Don't try to show plots if it failed badly

        
        finally:
            # Re-enable buttons
            self.run_button.setEnabled(True)
            self.upload_button.setEnabled(True)
            self.algorithm_combo.setEnabled(True)
            QApplication.processEvents() # Update UI

            # Show results/error window
            results_win = ResultsWindow(
                output_text,
                parent=self,
                metrics_text=metrics_text,
                selected_algorithm=selected_display_name,
                algorithm_output_dir=algo_output_dir, # Will be None if error occurred early
                next_day_prediction=next_day_prediction_result, # Pass the prediction
                is_error=error_occurred
            )
            results_win.exec()

          

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
