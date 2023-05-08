# NoduleNet-Nifty


## Notes about issues I had while installing and getting it to work for Nifty files. 

 

Created a new environment called env_NoduleNet which I will use for the NoduleNet in order to have all the correct dependencies. It has Python version 3.7 as this was needed for openCV download 

 

Struggled with the opencv install for a while and then I installed it using pip and it worked.  

Now I am struggling with an error message during setup: 
error: command 'gcc' failed with exit status 1 

 

 

 

I realized that the gcc error was actually caused by errors higher upp in the error message.  

First thing I did was install the ninja package and sencondly I changed the function AT_CHECK to TORCH_CHECK which is apparently needed as this package is not in newer python versions. The printed message when installing the script was still long and seemed to contain errors still. But it might work now.  

 

Then I had some issues with downloading the dataset they suggested. However it worked to download it using 7za x subsetX.zip. Downloading the data takes a lot of time so I will keep working on this tomorrow after the downloads are finished.  

 

 

 

There was an issue with following the preprocessing steps suggested in the NoduleNet readme file. However it was solved by installing the pynrrd package followd by changing the cvrt_annons_to_npy.py file to allow pickling of the data:  

> ctr_arrs = np.load(os.path.join(ctr_arr_dir, '%s.npy' % (pid)), allow_pickle=True) 

After this the scripts ran through the preprocessing.  

 

 

 

For the NoduleNet an issue I ran into was an error message that read: 

RuntimeError: No CUDA GPUs are available 

To solve it I had to change the following requirement to say 0 at the end (instead of 2): 

> os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

 

 

 

 

Managed to get NoduleNet to work on the LIDC data following the steps that they suggested for the preprocessing and dataselection.  

 

I used the dataset: 

> 'test_set_name': 'split/cross_val/0_val.csv' 

 

For the testing and the reported values from it that I got were 

> FROC at points:  [0.125, 0.25, 0.5, 1, 2, 4, 8] 

> fps:  0.125 , sensitivity:  0.6914893617021277 

> fps:  0.25 , sensitivity:  0.7819148936170213 

> fps:  0.5 , sensitivity:  0.8457446808510638 

> fps:  1 , sensitivity:  0.898936170212766 

> fps:  2 , sensitivity:  0.925531914893617 

> fps:  4 , sensitivity:  0.973404255319149 

> fps:  8 , sensitivity:  0.973404255319149 

> ============================================= 

> average FROC:  0.8700607902735563 

 

 

 

Talked to Luis about the lung package work on the Nodule Net. He said I should try to change the dataloading script instead of changing the data to .raw and .mhd data. He said there might be some preprocessing done before the data was converted into the .raw dataformat so I need to check how it looks after reading it in the script.  

 

Loading the data using SimpleITK worked if I loaded the .mhd file. The data does not look cropped or normalized. 

Looking at the data loaded in the test.py script, however, it seems like the data has been processed somehow. The data seems to have been normalized between –1 and 1 and there also seems like there has been HU clipping which I am not sure if it was done correclty. The image does not look to be cropped.  

 

 

I realized that the preprocessing of NoduleNet includes cropping the data around the lung mask as well as removing all data outside the lungs (setting it to pad_value = 170). The crop ends very close to lungs, risk of cropping part of the lungs? 

This means that I need to have segmentation of the lungs on order to use this model.  

 

 

 

Downloaded some of the files from Fidans data to test the model.  

Managed to get the script to run for the data, both preprocessing and eval. But when the eval script is run then it says in the end that there is no ground truth mask.  

 

It seems the script is reading the evaluationScript/annotations/LIDC/3_annotation.csv file, which seems to be a custom made file containing information about the location of the nodules. So It would appear as though this file is used to compare the predictions and as there are no files with these names in that file it assumes there are no gt annotations of nodules... 

 

 

 

Managed to make the preprocessing script and the evaluation script for NoduleNet to work for Nifty images! I made a script which creates a csv file for the preprocessed data based on the file called '_bboxes.npy' which consists of a tuple where each row contains [midpoint_x, midpoint_y, midpoint_z, diameter_max_xyz] for each of the nodules, if there are multiple nodule segmentations.  

 

I made a short script to guide through the preprocessing of Nifty files 

 

 
