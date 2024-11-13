import geocoder
import numpy as np

class Vehicle:
    """ Dummy class, simulating dronekit vehicle class. """
    def __init__(self):
        pass

    class location:
        def __init__(self):
            pass
        class global_frame:
            def __init__(self):
                self.lat = 55.752121 # Широта, град
                self.lon = 37.617664 # Долгота, град
                self.alt = 15 # Высота, м

    class attitude:
        def __init__(self):
            self.roll = 0 # Тангаж
            self.pitch = 0 # Крен
            self.yaw = 0 # Рысканье

vehicle = Vehicle

class DataProcessor:
    def __init__(self):
        # self.vehicle = connect('/dev/ttyUSB0', baud=57600, wait_ready=True) # Example of connection using dronekit
        # self.vehicle = vehicle()
        # self.gps = vehicle.location.global_frame
        pass

    def get_coordinates_from_vehicle(self, vehicle=vehicle):
        gps = vehicle.location.global_frame
        coordinates = gps.lat, gps.lon
        orientation = vehicle.attitude
        if coordinates is not None:
            return (coordinates, orientation)
        return ((0,0), (0,0,0))

    def get_coordinates_over_ip(self):
        coordinates = self.get_current_gps_coordinates()
        orientation = (0,0,0)
        if coordinates is not None:
            return (coordinates, orientation)
        return ((0,0), (0,0,0))

# TODO add dependence on roll and pitch
    def process_data(self, detection, coordinates, orientation):
        roll, pitch, azimuth = orientation
        dxpx, dypx = detection['center_offset']
        width, height = detection['frame_resolution']
        distance = detection['distance']
        dx = distance * dxpx / width
        dy = distance * dypx / height

        v1 = np.array([dx, dy])
        c, s = np.cos(azimuth), np.sin(azimuth)
        R = np.array(((c, -s), (s, c)))
        v2 = np.matmul(R, v1)
        x, y = int(v2[0]), int(v2[1])

        latitude, longitude = coordinates

        latitude = x + latitude
        longitude = y + longitude

        return latitude, longitude

    def export_data(self, detections):
        detection_data = []
        for detection in detections:
            latitude, longitude = self.process_data(detection, *self.get_coordinates_over_ip())
            record = {"unix_timestamp":detection['unix_timestamp'],
                      "latitude":latitude,
                      "longitude":longitude,
                      "probability":detection['probability']}
            detection_data.append(record)
        return detection_data

    def get_current_gps_coordinates(self):
        g = geocoder.ip('me') # Find the current information using IP
        if g.latlng is not None: # g.latlng tells if the coordiates are found or not
            return g.latlng
        return None
        

    def calculate_polar_coordinates(self, x, y):
        """ Calculate polar coordinates (radius, theta) relative to the image center. """
        theta = np.arctan2(-y, x)  # Angle in radians
        radius = np.sqrt(x**2 + y**2)  # Euclidean distance in pixels
        return radius, theta  