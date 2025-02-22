# RouteMaster 

Integrated real-time data analytics for optimized public transport, innovative road monitoring using demand prediction, and conditioning tech for sustainability, real-time pothole detection either by image or video, and smart parking count system for efficiency using AI/ML.

## Features

- **Demand Prediction**
- **Pothole Detection**
- **Real-time Road Damage Detection**
- **Optimize Route**
- **Cross-platform**

## Installation

To install and run this project, follow the steps below:

### 1. Clone the repo:

```bash
git clone https://github.com/Satyam-Mishra-1/RouteMaster
```

### 2. Navigate into the project directory:
```bash
cd RouteMaster
```

### 3. Set up the virtual environment with the required Python version:
For Windows:

# Download and install Python 3.11.9 (if you don't have it)
# Download link: https://www.python.org/downloads/release/python-3119/

# Create a virtual environment
python3.11 -m venv venv

# Activate the virtual environment
```
venv\Scripts\activate
```
For macOS/Linux:

#### Download and install Python 3.11.9 (if you don't have it)
#### Download link: https://www.python.org/downloads/release/python-3119/

# Create a virtual environment
```
python3.11 -m venv venv
```

# Activate the virtual environment
source venv/bin/activate
### 4. Install dependencies:
Make sure you have the correct virtual environment activated, and then run:

```
pip install -r requirements.txt
```

### 5. Run the project:
```
streamlit run app.py
```
You can now view the app in your browser:

Local URL: http://localhost:8501
To close the server, press ctrl + c.

Usage/Examples
After completing the installation steps, you can use the app for different functionalities.

How to use the app?
For Demand Prediction:
Navigate to the Demand Prediction section.
Click Browse Files.
Upload a .csv file. (For testing, a sample .csv file is included in the directory, named train_revised.csv.)
After uploading the .csv, the app will display the Home page of the Demand Prediction section below the video section.
You can select the appropriate demand page from the Choose Demand Page dropdown section.


For Road Damage Assessment:
Navigate to the Image Section.
Click Browse File.
Upload a JPG or JPEG file. (For testing, a sample .jpg file is included in the directory, named Pothole.jpg.)
After uploading the file, the model will process it and provide appropriate output.

The Route Section helps users plan the most efficient travel routes by solving the Traveling Salesman Problem (TSP). It calculates the shortest path between entered locations using geodesic distance and displays the route along with the distances between locations.

Simply enter the locations (e.g., city names), and the app will calculate the optimal route. A Google Maps link will also be provided for real-time navigation.



Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
