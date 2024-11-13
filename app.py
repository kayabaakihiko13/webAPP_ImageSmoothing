from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from matplotlib import pyplot as plt
from utils.image_smoothing import averange_image_3channel, gaussian_image_3channel, calculate_optimal_kernel_size
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import uuid

MEDIA_ROOT = "media"
# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = f"{MEDIA_ROOT}/uploads"
app.config["PROCESSED_FOLDER"] = f"{MEDIA_ROOT}/processed"

# Ensure folders exist
os.makedirs(MEDIA_ROOT, exist_ok=True)
os.makedirs(f"{MEDIA_ROOT}/uploads", exist_ok=True)
os.makedirs(f"{MEDIA_ROOT}/processed", exist_ok=True)

# Serve media files
@app.route('/media/<path:filename>')
def serve_media(filename):
    return send_from_directory(MEDIA_ROOT, filename)

def __saving_image_processed(image_arr: np.ndarray, filename: str) -> str:
    processed_filename = f"{uuid.uuid4().hex}_{filename}.png"
    processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    plt.imsave(processed_filepath, image_arr)
    print(f"Processed image saved at: {processed_filepath}")  # Debugging output
    return processed_filepath


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            # Secure and randomize the uploaded file name
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load and process image
            image = cv2.imread(filepath)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Determine kernel size
            kernel_option = request.form.get("kernel_option")
            kernel_size = 5 if kernel_option == "fixed" else calculate_optimal_kernel_size(image_rgb)
            
            # Retrieve algorithm option
            algorithm_option = request.form.get("algorithm_option")
            print(f"Selected algorithm: {algorithm_option}, Kernel size: {kernel_size}")  # Debugging output

            # Initialize processed_image_path
            processed_image_path = None
            print(f"Selected algorithm: {algorithm_option}")  # Debugging output

            # Apply selected filter and save the processed image
            if algorithm_option.lower() == "average_smoothing":
                print("Applying average smoothing...")
                average_filtered = averange_image_3channel(image_rgb, kernel_size=kernel_size)
                processed_image_path = __saving_image_processed(average_filtered, "average_image")
            elif algorithm_option.lower() == "gaussian_smoothing":
                print("Applying gaussian smoothing...")
                gaussian_filtered = gaussian_image_3channel(image_rgb, kernel_size=kernel_size)
                processed_image_path = __saving_image_processed(gaussian_filtered, "gaussian_image")
            else:
                print(f"Invalid algorithm option: {algorithm_option}")

            # Verify if processed_image_path was set
            if processed_image_path:
                processed_image_url = url_for('serve_media', filename=f'processed/{os.path.basename(processed_image_path)}')
            else:
                processed_image_url = None
                print("No processed image available.")  # Debugging output

            # Pass variables to the template
            return render_template(
                "display.html",
                original_image=url_for('serve_media', filename=f'uploads/{filename}'),
                processed_image=processed_image_url,
                filter_type=algorithm_option,
                kernel_size=kernel_size
            )

    # Render upload form if GET request
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
