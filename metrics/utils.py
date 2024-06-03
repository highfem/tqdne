import numpy as np 
import scipy.io
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
from scipy.signal import resample

class MatFileHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.mat_dict = None

    def read_mat_file(self):
        """
        Reads a MATLAB (.mat) file and stores its contents as a dictionary.
        """
        try:
            mat_contents = scipy.io.loadmat(self.file_path, struct_as_record=False, squeeze_me=True)
            self.mat_dict = self.mat_to_dict(mat_contents)
        except Exception as e:
            print(f"Error reading .mat file: {e}")

    def mat_to_dict(self, mat_obj):
        """
        Recursively converts a MATLAB structure to a nested dictionary.
        """
        if isinstance(mat_obj, dict):
            return {key: self.mat_to_dict(value) for key, value in mat_obj.items() if not (key.startswith('__') and key.endswith('__'))}
        elif isinstance(mat_obj, np.ndarray):
            if mat_obj.size == 1:
                return self.mat_to_dict(mat_obj.item())
            else:
                return [self.mat_to_dict(element) for element in mat_obj]
        elif hasattr(mat_obj, '_fieldnames'):
            return {field: self.mat_to_dict(getattr(mat_obj, field)) for field in mat_obj._fieldnames}
        else:
            return mat_obj

    def print_mat_structure(self, mat_obj=None, indent=0):
        """
        Recursively prints the structure of the MATLAB file contents.
        """
        if mat_obj is None:
            mat_obj = self.mat_dict

        if isinstance(mat_obj, dict):
            for key, value in mat_obj.items():
                print(' ' * indent + f"{key}:")
                self.print_mat_structure(value, indent + 4)
        elif isinstance(mat_obj, np.ndarray):
            print(' ' * indent + f"Array, Shape: {mat_obj.shape}, Dtype: {mat_obj.dtype}")
        elif hasattr(mat_obj, '_fieldnames'):
            print(' ' * indent + "MATLAB Object")
            for field in mat_obj._fieldnames:
                print(' ' * (indent + 4) + f"{field}:")
                self.print_mat_structure(getattr(mat_obj, field), indent + 8)
        else:
            print(' ' * indent + f"Type: {type(mat_obj)}")

    def process_data(self):
        """
        Processes specific data fields from the MATLAB file.
        """
        if self.mat_dict is None:
            print("MAT file not read yet.")
            return None, None

        try:
            rhyp = np.array(self.mat_dict['eq']['gan']['rhyp'])
            vs30 = np.array(self.mat_dict['eq']['gan']['vs30'])
            idx = np.where((vs30 > 0) & (~np.isnan(vs30)))

            rhyp = rhyp[idx]
            vs30 = vs30[idx]

            wf = np.array(self.mat_dict['eq']['gan']['wfMat'])

            return rhyp, vs30, wf
        except KeyError as e:
            print(f"Key error: {e}")
            return None, None, None
        

def shakeMap_cscale(mmi=None):
    """
    Returns a new ShakeMap MMI colormap for any vector of mmi values.
    Without input arguments, it returns the standard scale colormap.
    
    :param mmi: List or array of MMI values.
    :return: A matplotlib colormap object.
    
    # Example usage:
    cscale = shakeMap_cscale()

    # Generate some example data
    np.random.seed(0)
    x = np.random.uniform(0, 10, 100)
    y = np.random.uniform(0, 10, 100)
    z = np.random.uniform(1, 10, 100)  # These will be the MMI values

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=z, cmap=cscale, edgecolor='k')
    plt.colorbar(sc, label='MMI', ticks=np.arange(1, 11))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot with ShakeMap MMI Colormap')
    plt.grid(True)
    plt.show()
    """
    
    if mmi is None:
        mmi = np.linspace(1, 10, 256)
    
    num_colors = len(mmi)
    
    # The colors in the original color map are the colors of the 11 edges of
    # the 10 bins, evenly spaced from 0.5 to 10.5.
    color_map = np.array([
        [255, 255, 255],
        [191, 204, 255],
        [160, 230, 255],
        [128, 255, 255],
        [122, 255, 147],
        [255, 255, 0],
        [255, 200, 0],
        [255, 145, 0],
        [255, 0, 0],
        [200, 0, 0],
        [128, 0, 0]
    ]) / 255.0

    mmi_values = np.arange(1, 12)
    
    colormap_data = np.zeros((num_colors, 3))
    for i in range(3):
        interpolator = interp1d(mmi_values, color_map[:, i], kind='linear')
        colormap_data[:, i] = interpolator(mmi)
    
    # Create a colormap
    colormap = LinearSegmentedColormap.from_list('ShakeMapMMI', colormap_data, N=num_colors)
    
    return colormap


def pga_to_mmi(pga, unit='g'):
    """
    Convert Peak Ground Acceleration (PGA) to Modified Mercalli Intensity (MMI).

    Parameters:
    pga (float or numpy array): Peak Ground Acceleration.
    unit (str): Unit of the PGA ('g' for gravity or 'm/s^2' for meters per second squared).

    Returns:
    float or numpy array: Modified Mercalli Intensity (MMI).
    """
    # Ensure PGA is in the form of a numpy array for consistent operations
    pga = np.asarray(pga)
    
    # Conversion factor from m/s^2 to g
    if unit == 'm/s^2':
        pga = pga / 9.80665  # 1 g = 9.80665 m/s^2
    elif unit == 'cm/s^2':
        pga = pga / 9.80665 *1e-2 # 1 g = 9.80665 m/s^2
    
    # Apply the empirical formula
    mmi = 3.66 * np.log10(pga) + 1.66
    
    return mmi

def calculate_gmrotd50(component1, component2):
    """
    Calculate the GMRotD50 from two horizontal component seismograms.

    Parameters:
    component1 (np.ndarray): Seismogram of the first horizontal component.
    component2 (np.ndarray): Seismogram of the second horizontal component.

    Returns:
    gmrotd50 (np.ndarray): The GMRotD50 values.
    """
    len1 = len(component1)
    len2 = len(component2)
    
    if len1 != len2:
        # Resample the shorter seismogram to match the length of the longer one
        if len1 < len2:
            component1 = resample(component1, len2)
        else:
            component2 = resample(component2, len1)

    # Number of rotation angles
    num_angles = 180
    gmrotd_values = np.zeros((num_angles, len(component1)))

    # Compute GMRotD for each rotation angle
    for angle in range(num_angles):
        theta = np.deg2rad(angle)
        rotated1 = component1 * np.cos(theta) + component2 * np.sin(theta)
        rotated2 = -component1 * np.sin(theta) + component2 * np.cos(theta)
        gmrotd = np.sqrt(rotated1**2 + rotated2**2)
        gmrotd_values[angle, :] = gmrotd

    # Compute GMRotD50 as the 50th percentile of the geometric mean
    gmrotd50 = np.percentile(gmrotd_values, 50, axis=0)

    return np.max(gmrotd50)
