import cv2
import Image,ImageFilter,ImageEnhance
from sklearn import datasets,svm
from sklearn.externals import joblib
import pickle

import os,urllib,time

def preprocess(filepath):
	im = Image.open(filepath)
	im = im.convert('1')
	return im

def split(im):
	box = [(13,5,53,90),(59,5,99,90),(105,5,145,90),(151,5,191,90)]
	imbox = []
	for i in range(0,4):
		imbox.append(im.crop(box[i]))
	return imbox
	
def showim(im):
	im.save("tmp.png")
	img = cv2.imread("tmp.png")
	cv2.namedWindow("Image")
	cv2.imshow("Image",img)
	key = cv2.waitKey(0)
	cv2.destroyAllWindows()
	return key

def im2matrix(im):
	t=[]
	for x in xrange(im.size[0]):
		tt=0
		for y in xrange(im.size[1]):
			if im.getpixel((x,y)) == 0:
				tt += 1
		t.append(tt)
	return t

def learnimg(filepath):
	im = preprocess(filepath)
	xbox = split(im)
	data = []
	key = []
	for x in xbox:
		key.append(showim(x) - 0x30)
		data.append(im2matrix(x))
	return data,key

def learn():
	clf = svm.SVC(gamma=1, C=100.,probability=True,kernel='linear')
	data=[]
	key=[]
	for i in range(1,15):
		da,ke = learnimg('downloadcode/%d.jpg'%i)
		for i in range(0,4):
			data.append(da[i])
			key.append(ke[i])
	clf.fit(data,key) 	
	s = pickle.dumps(clf)
	f = open('learn.pkl','w+')
	f.write(s)
	f.close()

def test(filename,clf):
	im = preprocess(filename)
	imbox = split(im)

	result = ''
	for i in range(0,4):
		result = result+str(clf.predict(im2matrix(imbox[i]))[0])

	return int(result)


##############################################################################

def downloadimg(name):
	imgURL = "http://xk.urp.seu.edu.cn/studentService/getCheckCode"
	dist = os.path.join("", "test/%s.jpg"%name)
	urllib.urlretrieve(imgURL, dist,None) 

t = [
2092,7679,2644,8565,2596,8752,
7823,6929,7583,7560,6377,3034,
6452,7307,5053,8462,6699,4967,
5603,590,6829,8920,2369,6405,
5794,393,4552,7359,3595,6394,
9599,5689,8006,3337,2860,4092,
9749,2690,5394,495,3922,268,
6396,7705,5393,2559,5525,3580,
38,402,8065,650,353,8294,
6046,7802,5327,200,4422,5875,
2769,5842,5332,462,8445,4605,
7777,3098,8367,5283,4242,8990,
3902,2240,760,2360,3934,6944,
5297,8220,733,4534,4775,4492,
9455,6288,2589,2647,4906,6597,
9987,3839,2592,8587,4298,7239,
9622,8655,8265,4496,6876,9374,
633,3556,2673,6788,9336,8825,
9624,887,7957,2634,7363,9775,
8450,6239,6322,539,488,490,
627,502,5576,4847,7296,8558,
3688,6374,7282,326,8982,900,
6980,855,2793,3678,4744,9223,
8947,3090,4940,9547,4064,4542,
7362,3096,6837,2358,6758,75,
2552,526,7833,7579,887,2968,
3880,4699,4884,8984,975,397,
3668,3282,3723,3557,6090,7398,
3683,439,9688,3990,3335,8552,
4059,5900,8932,7775,6599,2759,#180
]

#learn()
#for i in range(1,500):
#	downloadimg("%d"%i)
#	time.sleep(1)

def identity():
	#learn()

	f = open('learn.pkl','r')
	s = f.read()
	f.close()
	clf = pickle.loads(s)
	
	ri = 0
	for i in range(0,180):
		r = test('test/%d.jpg'%(i+1),clf)
		if t[i] != r :
			print t[i],r,i+1
			ri = ri + 1
	print "Rate:",1-float(ri)/180

identity()