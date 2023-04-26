# This module tries to match the found endpoints to satellite positions from known TLEs.

import os
import sys
import numpy as np
import pickle as pickle

sys.path.append("/net/beulakerwijde/data1/breslin/code/fotos-python3/")

class CheckKnownSatellites(object):
    def __init__(self, passages, user, target, rootdir):

        # Dictionary that contains the known satellites that were in FoV
        self.passages = passages

        # Dict to save found sats that are already known i.e. end-points are matched to expected sky position from TLE
        self.found_satellites = {'negative':{}, 'positive':{}}

        # Dict in which we (separately) save the actual pixel positions of sats that we find
        self.pixel_positions = {'negative':{}, 'positive':{}}

        # These user and target are used for directory and file names when storing the dictionaries
        self.target = target
        self.dirname = f'{rootdir}{user}/data/subtracted/{target}/SkyPositions/'

        if os.path.exists(f'{self.dirname}known_satellites_{target}.p'):
            choice = input("Dictionaries have already been created in this directory! Continue? [y/n] ").lower()
            if choice == "y":
                pass
            else:
                sys.exit()

    def match_start_endpoints(self, JD, x_min, y_min, x_max, y_max, sat_vmag, lst, exposure, ransac_model, astro, 
        midlst, midJD, lstseq, keyword=None, threshold=10, save_to_file=True):

        """     
        - Check if the found start- and end-points are close to expected positions of known satellites i.e. within a 
            number of pixels set by the threshold parameter.
        - JD = Julian Day of the image in which the track was found.
        - x_min, y_min, x_max, y_max = pixel position of end-points of track.
        - sat_vmag = visual magnitude of the satellite (determined from nearby stars).
        - lst = local sidereal time of the image in which the track was found.
        - exposure = exposure time of the image (in seconds).
        - ransac_model = the Ransac model that was found to fit the track.
        - astro = the astrometric solution, needed to convert pixel positions to ra and dec (and vice versa).
        - midlist = the lst of the image to which current image was shifted and stacked in the subtract.py routine. 
            Needed for small corrections in astrometric solution.
        - midJD = the JD of the image to which current image was shifted and stacked in the subtract.py routine. 
            Needed for small corrections in astrometric solution.
        - lstseq = is the lst sequence number with which the image can be identified (file name)
        - keyword: should be either 'negative' or 'positive' to refer to the image in which the track was found. 
            Otherwise tracks are saved ones, now we save the track twice if it is found in both neg and pos pixel value.
            Remember than each image is used twice in the consecutive subtract routine.
        - threshold = the maximum number of pixels that the end-points can differ from the TLE and SGP4 derived values
        - save_to_file = whether or not the save sky and pixel positions to a dictionary on the hard drive. 

        """ 


        for sat in list(self.passages[lstseq]):

            # Since it's not clear which of the end-points is the start and which is the end, we have to check both:

            differences1 = (
                self.passages[lstseq][sat]['start']['x0'] - x_min, 
                self.passages[lstseq][sat]['start']['y0'] - y_min, 
                self.passages[lstseq][sat]['end']['x0'] - x_max, 
                self.passages[lstseq][sat]['end']['y0'] - y_max
                )

            differences2 = (
                self.passages[lstseq][sat]['start']['x0'] - x_max, 
                self.passages[lstseq][sat]['start']['y0'] - y_max, 
                self.passages[lstseq][sat]['end']['x0'] - x_min, 
                self.passages[lstseq][sat]['end']['y0'] - y_min
                )


            # Check if both the start- and end-point are within the threshold number of pixels of a known satellite
            if (all(abs(x)< threshold for x in differences1)):

                # Create new entry if satellite has not been observed before                  
                if sat not in list(self.found_satellites[keyword].keys()):

                    # Save the RA and DEC and corresponding LST for start- and end-point in a dictionary. 
                    # Use SAT ID, and then subsequently the JD, as keys to save the RA, DEC and LST of each endpoint, 
                    # and the visual magnitude of the satellite.
                    # NOTE that currently JD and LST -/+ 0.5 exposure time are saved!!! Is this accurate timing?
                    # Saved in the format:  {JD:[RA, DEC, lst], 'vmag':{JD1:sat_vmag1, JD2:sat_vmag2, etc.}, 
                    # 'pixelpositions':{JD:{'FOTOS:[x,y], 'SGP4':[x,y], 'lstseq':lstseq}}

                    self.found_satellites[keyword][sat] = {

                        self.passages[lstseq][sat]['start']['jd']:[np.squeeze(astro.pix2world(midlst, [x_min], [y_min], 
                            jd=midJD)), self.passages[lstseq][sat]['start']['lst']],
                        self.passages[lstseq][sat]['end']['jd']:[np.squeeze(astro.pix2world(midlst, [x_max], [y_max], 
                            jd=midJD)), self.passages[lstseq][sat]['end']['lst']],

                        'vmag':{lstseq:sat_vmag},
                        'pixelpositions':{

                            self.passages[lstseq][sat]['start']['jd']:{
                                'FOTOS':[x_min, y_min],
                                'SGP4':[self.passages[lstseq][sat]['start']['x0'], 
                                self.passages[lstseq][sat]['start']['y0']],
                                'lstseq':lstseq
                                },

                            self.passages[lstseq][sat]['end']['jd']:{
                                'FOTOS':[x_max, y_max],
                                'SGP4':[self.passages[lstseq][sat]['end']['x0'],
                                self.passages[lstseq][sat]['end']['y0']],
                                'lstseq':lstseq
                                }
                            }
                        }

                    # Also save the pixel positions in a separate dictionary
                    self.pixel_positions[keyword][sat]={lstseq:{'start':[x_min, y_min], 'end':[x_max, y_max]}}

  
                else: # add to existing satellite entry 

                    self.found_satellites[keyword][sat][self.passages[lstseq][sat]['start']['jd']] = \
                        [np.squeeze(astro.pix2world(midlst, [x_min], [y_min], jd=midJD)), \
                            self.passages[lstseq][sat]['start']['lst']]

                    self.found_satellites[keyword][sat][self.passages[lstseq][sat]['end']['jd']] = \
                        [np.squeeze(astro.pix2world(midlst, [x_max], [y_max], jd=midJD)), \
                            self.passages[lstseq][sat]['end']['lst']]

                    self.found_satellites[keyword][sat]['vmag'].update({lstseq:sat_vmag})
                    self.found_satellites[keyword][sat]['pixelpositions'].update({

                        self.passages[lstseq][sat]['start']['jd']:{
                            'FOTOS':[x_min, y_min], 
                            'SGP4':[self.passages[lstseq][sat]['start']['x0'],
                            self.passages[lstseq][sat]['start']['y0']],
                            'lstseq':lstseq
                            },

                        self.passages[lstseq][sat]['end']['jd']:{
                            'FOTOS':[x_max, y_max],
                            'SGP4':[self.passages[lstseq][sat]['end']['x0'],
                            self.passages[lstseq][sat]['end']['y0']],
                            'lstseq':lstseq
                            }
                        })

                    # Also save the pixel positions 
                    self.pixel_positions[keyword][sat][lstseq] = {'start': [x_min, y_min], 'end': [x_max, y_max]}


            # Same routine as above, but start- and end-point are 'reversed'                            
            elif (all(abs(x)< 10 for x in differences2)):

                # Create new entry
                if sat not in list(self.found_satellites[keyword].keys()):

                    self.found_satellites[keyword][sat] = {

                        self.passages[lstseq][sat]['start']['jd']:[np.squeeze(astro.pix2world(midlst, [x_max], [y_max], 
                            jd=midJD)), self.passages[lstseq][sat]['start']['lst']],
                        self.passages[lstseq][sat]['end']['jd']:[np.squeeze(astro.pix2world(midlst, [x_min], [y_min], 
                            jd=midJD)), self.passages[lstseq][sat]['end']['lst']],

                        'vmag':{lstseq:sat_vmag},
                        'pixelpositions':{

                            self.passages[lstseq][sat]['start']['jd']:{
                                'FOTOS':[x_max, y_max],
                                'SGP4':[self.passages[lstseq][sat]['start']['x0'],
                                self.passages[lstseq][sat]['start']['y0']],
                                'lstseq':lstseq
                                },

                            self.passages[lstseq][sat]['end']['jd']:{
                                'FOTOS':[x_min, y_min],
                                'SGP4':[self.passages[lstseq][sat]['end']['x0'],
                                self.passages[lstseq][sat]['end']['y0']],
                                'lstseq':lstseq
                                }
                            }
                        }

                    # Also save the pixel positions 
                    self.pixel_positions[keyword][sat]={lstseq:{'start':[x_max, y_max], 'end':[x_min, y_min]}}


                else: # Add to existing satellite entry 

                    self.found_satellites[keyword][sat][self.passages[lstseq][sat]['start']['jd']] = \
                        [np.squeeze(astro.pix2world(midlst, [x_max], [y_max], jd=midJD)), 
                        self.passages[lstseq][sat]['start']['lst']]
                    self.found_satellites[keyword][sat][self.passages[lstseq][sat]['end']['jd']] = \
                        [np.squeeze(astro.pix2world(midlst, [x_min], [y_min], jd=midJD)), 
                        self.passages[lstseq][sat]['end']['lst']]

                    self.found_satellites[keyword][sat]['vmag'].update({lstseq:sat_vmag})
                    self.found_satellites[keyword][sat]['pixelpositions'].update({

                        self.passages[lstseq][sat]['start']['jd']:{
                            'FOTOS':[x_max, y_max],
                            'SGP4':[self.passages[lstseq][sat]['start']['x0'],
                            self.passages[lstseq][sat]['start']['y0']],
                            'lstseq':lstseq
                            },

                        self.passages[lstseq][sat]['end']['jd']:{
                            'FOTOS':[x_min, y_min],
                            'SGP4':[self.passages[lstseq][sat]['end']['x0'],
                            self.passages[lstseq][sat]['end']['y0']],
                            'lstseq':lstseq
                            }
                        })

                    # Also save the pixel positions 
                    self.pixel_positions[keyword][sat][lstseq] = {'start': [x_max, y_max], 'end':[x_min, y_min]}

        # Save the dictionaries
        if save_to_file is True:
            pickle.dump(self.found_satellites, open(f"{self.dirname}found_satellites_{lstseq}.p", "wb" ))
            pickle.dump(self.pixel_positions, open(f"{self.dirname}pixel_positions_{lstseq}.p", "wb" ))
                                                                                                                   