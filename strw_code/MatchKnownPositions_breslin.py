#This module tries to match the found endpoints to satellite positions from known TLEs.

import numpy as np
import pickle as pickle
import os
import sys
sys.path.append("/net/beulakerwijde/data1/breslin/code/fotos-python3/")

class CheckKnownSatellites(object):
    def __init__(self, passages, user, target, rootdir):
        #Dictionary that contains the known satellites that were in FoV
        self.passages = passages
        #Dictionary in which we save found satellites that are already known (i.e. endpoints are matched to expected sky position from TLE)
        #Saved in the format:  {JD:[RA, DEC, lst], 'vmag':{JD1:sat_vmag1, JD2:sat_vmag2, etc.}, 'pixelpositions':{JD:{'FOTOS:[x,y], 'SGP4':[x,y], 'lstseq':lstseq}}
        self.found_satellites = {'negative':{}, 'positive':{}}
        #Dictionary in which we (separately) save the actual pixel positions of satellites that we find
        #Saved in the format:  {sat_id:{JD:{'start': [x_min, y_min], 'end': [x_max, y_max]}}}
        #DO WE WANT TO SAVE THE PIXEL POSITIONS? AND IF SO, DO WE WANT TO SAVE THE JD OF BOTH THE START AND ENDPOINT? (AND NOT ONLY OF IMAGE WHICH IS CURRENTLY THE CASE)
        self.pixel_position_found_satellites = {'negative':{}, 'positive':{}}

        #These user and target are used for directory and file names when storing the dictionaries
        self.target = target
        self.dirname = f'{rootdir}{user}/data/subtracted/{target}/SkyPositions/'

        # if os.path.exists(f'{self.dirname}known_satellites_{target}.p'):

        #     choice = input("Warning! Dictionaries have already been created in this directory! Do you want to continue? [y/n]").lower()

        #     if choice == 'y':
        #         pass
        #     else:
        #         sys.exit()

    def match_start_endpoints(self,JD, x_min, y_min, x_max, y_max, sat_vmag, lst, exposure, ransac_model, astro, midlst, midJD, lstseq, keyword=None, threshold=10, save_to_file=True):
        #Check if the found start- and endpoints are close to expected positions of known satellites (within a number of pixels set by the threshold parameter)
        #JD = Julian Day of the image in which the track was found
        #x_min, y_min, x_max, y_max =pixel position of end points of track
        #sat_vmag = visual magnitude of the satellite (determined from nearby stars)
        #lst = local sidereal time of the image in which the track was found
        #exposure = exposure time of the image (in seconds)
        #ransac_model = the Ransac model that was found to fit the track
        #astro = the astrometric solution, needed to convert pixel positions to ra and dec (and vice versa)
        #midlist = is the lst of the image to which current image was shifted and stacked in the subtract.py routine. Needed for small corrections in astrometric solution 
        #midJD = is the JD of the image to which current image was shifted and stacked in the subtract.py routine. Needed for small corrections in astrometric solution
        #lstseq = is the lst sequence number with which the image can be identified (file name)
        #keyword: should be either 'negative' or 'positive' to refer to the image in which the track was found. (otherwise tracks are saved ones, now we save the track twice if it is found in both negative and positive pixel value. Remember than each image is used twice in the consecutive subtract routine.) 
        #threshold = is the maximum number of pixels are found end points can differ from the TLE and SGP4 derived values
        #save_to_file = whether or not the save sky and pixel positions to a dictionary on the hard drive.  


        for sat in list(self.passages[JD].keys()):


            #Since it's not clear which of the endpoints is the start and which is the end, we have to check both possibilities
            differences1 = self.passages[JD][sat]['start']['x0'] - x_min, self.passages[JD][sat]['start']['y0'] - y_min, self.passages[JD][sat]['end']['x0'] - x_max, self.passages[JD][sat]['end']['y0'] - y_max

            differences2 = self.passages[JD][sat]['start']['x0'] - x_max, self.passages[JD][sat]['start']['y0'] - y_max, self.passages[JD][sat]['end']['x0'] - x_min, self.passages[JD][sat]['end']['y0'] - y_min


            if (all(abs(x)< threshold for x in differences1)):
                #Both start- and endpoint are within the threshold number of pixels of a known satellite

                #Create new entry if satellite has not been observed before                  
                if sat not in list(self.found_satellites[keyword].keys()):
                    #Save the RA and DEC and corresponding LST for start- and endpoint in a Python dictionary. Use SAT ID, and then subsequently the JD, as keys to save the RA, DEC and LST of each endpoint, and the visual magnitude of the satellite.
                    #NOTE that currently JD and LST -/+ 0.5 exposure time are saved!!! Is this accurate timing?
                    #Saved in the format:  {JD:[RA, DEC, lst], 'vmag':{JD1:sat_vmag1, JD2:sat_vmag2, etc.}, 'pixelpositions':{JD:{'FOTOS:[x,y], 'SGP4':[x,y], 'lstseq':lstseq}}
                    self.found_satellites[keyword][sat] = {self.passages[JD][sat]['start']['jd']:[np.squeeze(astro.pix2world(midlst, [x_min], [y_min], jd=midJD)), self.passages[JD][sat]['start']['lst']],
                                                  self.passages[JD][sat]['end']['jd']:[np.squeeze(astro.pix2world(midlst, [x_max], [y_max], jd=midJD)), self.passages[JD][sat]['end']['lst']],
                                                  'vmag':{JD:sat_vmag},
                                                  'pixelpositions':{self.passages[JD][sat]['start']['jd']:{'FOTOS':[x_min, y_min],
                                                                                                           'SGP4':[self.passages[JD][sat]['start']['x0'],
                                                                                                                   self.passages[JD][sat]['start']['y0']],
                                                                                                           'lstseq':lstseq},
                                                                    self.passages[JD][sat]['end']['jd']:{'FOTOS':[x_max, y_max],
                                                                                                         'SGP4':[self.passages[JD][sat]['end']['x0'],
                                                                                                                 self.passages[JD][sat]['end']['y0']],
                                                                                                         'lstseq':lstseq}
                                                                    }
                                                  }
                    #Also save the pixel positions in a separate dictionary
                    self.pixel_position_found_satellites[keyword][sat]={JD:{'start':[x_min, y_min], 'end':[x_max, y_max]}}


                    #Here the positions from the TLE are stored, not the endpoints we've determined with our routines
                    #found_satellites[sat] = {passages[JD][sat]['start']['jd']:[[passages[JD][sat]['start']['ra'], passages[JD][sat]['start']['dec']], passages[JD][sat]['start']['lst']]}#, passages[JD][sat]['end']['jd']:[[passages[JD][sat]['end']['ra'], passages[JD][sat]['end']['dec']], passages[JD][sat]['end']['lst']]}                        

                #Else, add to existing satellite entry   
                else:

                    self.found_satellites[keyword][sat][self.passages[JD][sat]['start']['jd']] = [np.squeeze(astro.pix2world(midlst, [x_min], [y_min], jd=midJD)), self.passages[JD][sat]['start']['lst']]
                    self.found_satellites[keyword][sat][self.passages[JD][sat]['end']['jd']] = [np.squeeze(astro.pix2world(midlst, [x_max], [y_max], jd=midJD)), self.passages[JD][sat]['end']['lst']]
                    self.found_satellites[keyword][sat]['vmag'].update({JD:sat_vmag})
                    self.found_satellites[keyword][sat]['pixelpositions'].update({self.passages[JD][sat]['start']['jd']:{'FOTOS':[x_min, y_min],
                                                                                                                'SGP4':[self.passages[JD][sat]['start']['x0'],
                                                                                                                        self.passages[JD][sat]['start']['y0']],
                                                                                                                'lstseq':lstseq},
                                                                        self.passages[JD][sat]['end']['jd']:{'FOTOS':[x_max, y_max],
                                                                                                             'SGP4':[self.passages[JD][sat]['end']['x0'],
                                                                                                                     self.passages[JD][sat]['end']['y0']],
                                                                                                             'lstseq':lstseq}})

                    #Here the positions from the TLE are stored, not the endpoints we've determined with our routines
                    #found_satellites[sat][passages[JD][sat]['start']['jd']] = [[passages[JD][sat]['start']['ra'], passages[JD][sat]['start']['dec']], passages[JD][sat]['start']['lst']]
                    #found_satellites[sat][passages[JD][sat]['end']['jd']] = [[passages[JD][sat]['end']['ra'], passages[JD][sat]['end']['dec']], passages[JD][sat]['end']['lst']]

                    self.pixel_position_found_satellites[keyword][sat][JD] = {'start': [x_min, y_min], 'end': [x_max, y_max]}

            #Same routine as above, but start- and endpoint are 'reversed'                            
            else: 
            #elif (all(abs(x)< 10 for x in differences2)):

                #Create new entry
                if sat not in list(self.found_satellites[keyword].keys()):
                    self.found_satellites[keyword][sat] = {self.passages[JD][sat]['start']['jd']:[np.squeeze(astro.pix2world(midlst, [x_max], [y_max], jd=midJD)), self.passages[JD][sat]['start']['lst']],
                                                  self.passages[JD][sat]['end']['jd']:[np.squeeze(astro.pix2world(midlst, [x_min], [y_min], jd=midJD)), self.passages[JD][sat]['end']['lst']],
                                                  'vmag':{JD:sat_vmag},
                                                  'pixelpositions':{self.passages[JD][sat]['start']['jd']:{'FOTOS':[x_max, y_max],
                                                                                                           'SGP4':[self.passages[JD][sat]['start']['x0'],
                                                                                                                   self.passages[JD][sat]['start']['y0']],
                                                                                                           'lstseq':lstseq},
                                                                    self.passages[JD][sat]['end']['jd']:{'FOTOS':[x_min, y_min],
                                                                                                         'SGP4':[self.passages[JD][sat]['end']['x0'],
                                                                                                                 self.passages[JD][sat]['end']['y0']],
                                                                                                         'lstseq':lstseq}
                                                                    }
                                                  }

                    self.pixel_position_found_satellites[keyword][sat]={JD:{'start':[x_max, y_max], 'end':[x_min, y_min]}}

                    #Here the positions from the TLE are stored, not the endpoints we've determined with our routines
                    #found_satellites[sat] = {passages[JD][sat]['start']['jd']:[[passages[JD][sat]['start']['ra'], passages[JD][sat]['start']['dec']], passages[JD][sat]['start']['lst']]}#, passages[JD][sat]['end']['jd']:[[passages[JD][sat]['end']['ra'], passages[JD][sat]['end']['dec']], passages[JD][sat]['end']['lst']]}

                #Add to existing satellite entry 
                else:

                    self.found_satellites[keyword][sat][self.passages[JD][sat]['start']['jd']] = [np.squeeze(astro.pix2world(midlst, [x_max], [y_max], jd=midJD)), self.passages[JD][sat]['start']['lst']]
                    self.found_satellites[keyword][sat][self.passages[JD][sat]['end']['jd']] = [np.squeeze(astro.pix2world(midlst, [x_min], [y_min], jd=midJD)), self.passages[JD][sat]['end']['lst']]
                    self.found_satellites[keyword][sat]['vmag'].update({JD:sat_vmag})
                    self.found_satellites[keyword][sat]['pixelpositions'].update({self.passages[JD][sat]['start']['jd']:{'FOTOS':[x_max, y_max],
                                                                                                                'SGP4':[self.passages[JD][sat]['start']['x0'],
                                                                                                                        self.passages[JD][sat]['start']['y0']],
                                                                                                                'lstseq':lstseq},
                                                                        self.passages[JD][sat]['end']['jd']:{'FOTOS':[x_min, y_min],
                                                                                                             'SGP4':[self.passages[JD][sat]['end']['x0'],
                                                                                                                     self.passages[JD][sat]['end']['y0']],
                                                                                                             'lstseq':lstseq}})

                    self.pixel_position_found_satellites[keyword][sat][JD] = {'start': [x_max, y_max], 'end':[x_min, y_min]}

                    #Here the positions from the TLE are stored, not the endpoints we've determined with our routines
                    #found_satellites[sat][passages[JD][sat]['start']['jd']] = [[passages[JD][sat]['start']['ra'], passages[JD][sat]['start']['dec']], passages[JD][sat]['start']['lst']]
                    #found_satellites[sat][passages[JD][sat]['end']['jd']] = [[passages[JD][sat]['end']['ra'], passages[JD][sat]['end']['dec']], passages[JD][sat]['end']['lst']]



        if save_to_file is True:
            pickle.dump(self.found_satellites, open(f'{self.dirname}found_sats_{lstseq}.p', "wb" ))
            pickle.dump(self.pixel_position_found_satellites, open(f'{self.dirname}pxlpos_{lstseq}.p', "wb" ))
                                                                                                                   
