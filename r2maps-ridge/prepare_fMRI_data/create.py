files = [glob.glob(os.path.join(rootdir, "fmri-data/en", "sub-%03d" % s, "func","*.nii")) for s in subjects]
print(files[1][2][:-21])