from getpass import getpass
import os

# This downloads the raw hdf modis file used in the idaho fire succession model. You will need a NASA Earthdata
# login to download from the FTP (available for free)

download_dir = './data/modis'

directory_exists = os.path.isdir(download_dir)
if not directory_exists:
    os.makedirs(download_dir)

file_list=['https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.051/2001.01.01/MCD12Q1.A2001001.h09v04.051.2014287162024.hdf',
           'https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.051/2002.01.01/MCD12Q1.A2002001.h09v04.051.2014287170022.hdf',
           'https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.051/2003.01.01/MCD12Q1.A2003001.h09v04.051.2014287182211.hdf',
           'https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.051/2004.01.01/MCD12Q1.A2004001.h09v04.051.2014287174031.hdf',
           'https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.051/2005.01.01/MCD12Q1.A2005001.h09v04.051.2014287190241.hdf',
           'https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.051/2006.01.01/MCD12Q1.A2006001.h09v04.051.2014287194338.hdf',
           'https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.051/2007.01.01/MCD12Q1.A2007001.h09v04.051.2014287211100.hdf',
           'https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.051/2008.01.01/MCD12Q1.A2008001.h09v04.051.2014288150448.hdf',
           'https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.051/2009.01.01/MCD12Q1.A2009001.h09v04.051.2014288175117.hdf',
           'https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.051/2010.01.01/MCD12Q1.A2010001.h09v04.051.2014288182320.hdf',
           'https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.051/2011.01.01/MCD12Q1.A2011001.h09v04.051.2014288190332.hdf',
           'https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.051/2012.01.01/MCD12Q1.A2012001.h09v04.051.2014288200052.hdf',
           'https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.051/2013.01.01/MCD12Q1.A2013001.h09v04.051.2014308185258.hdf']


username = input('NASA EarthLogin username: ')
password  = getpass('NASA Earthlogin password: ')

total_files = len(file_list)
for i, this_file in enumerate(file_list,1):
    print('downloading '+str(i)+' of '+str(total_files))
    os.system('wget --user={0} --password={1} -P {2} {3}'.format(username, password, download_dir, this_file))
