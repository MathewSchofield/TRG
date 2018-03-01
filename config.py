run = '1000stars'

pins_floc = '/home/mxs191/Desktop/MathewSchofield/TRG/GetData/Pinsonneault2014.csv'

if run == '20stars':
    # the directories used in likeTESS1.py getInput()
    data_dir = '/home/mxs191/Desktop/MathewSchofield/TRG/GetData/20Stars/'
    param_file = data_dir + '20stars.csv'
    mag_file = data_dir + '20stars_simbad.csv'
    mode_dir = '/home/mxs191/Desktop/MathewSchofield/TRG/GetData/Modes/'

    # The directory to save files in DataTest1.py, data_for_ML() class, save_xy()
    ML_data_dir = '/home/mxs191/Desktop/MathewSchofield/TRG/DetTest/' +\
				  'DetTest1_results/data_for_ML/20Stars/20Stars'

if run == '100stars':
    data_dir = '/home/mxs191/Desktop/MathewSchofield/TRG/GetData/100Stars/'
    param_file = data_dir + '100stars.csv'
    mag_file = data_dir + '100stars_simbad.csv'
    mode_dir = '/home/mxs191/Desktop/MathewSchofield/TRG/GetData/Modes/'
    ML_data_dir = '/home/mxs191/Desktop/MathewSchofield/TRG/DetTest/' +\
				'DetTest1_results/data_for_ML/100Stars/100Stars'


if run == '1000stars':
    data_dir = '/home/mxs191/Desktop/MathewSchofield/TRG/GetData/1000Stars/'
    param_file = data_dir + '1000stars.csv'
    mag_file = data_dir + '1000stars_simbad4.csv'
    mode_dir = '/home/mxs191/Desktop/MathewSchofield/TRG/GetData/Modes/'
    ML_data_dir = '/home/mxs191/Desktop/MathewSchofield/TRG/DetTest/' +\
				'DetTest1_results/data_for_ML/1000Stars/1000Stars'

