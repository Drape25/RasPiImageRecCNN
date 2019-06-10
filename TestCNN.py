#precision = (true positives)/(true positives + false positives)
#recall = true positive/(true positive + false negative)
#accuracy = (true positive+true negative)/total

total = 0
totaltrupos = 0
totalfalsepos = 0
totalfalseneg = 0
totaltrueneg = 0

humantruepositive = 0
humanfalsepositive = 0
humanfalsenegative = 0
humantruenegative = 0

leopardtruepositive = 0
leopardfalsepositive = 0
leopardfalsenegative = 0
leopardtruenegative = 0

neithertruepositive = 0
neitherfalsepositive = 0
neitherfalsenegative = 0
neithertruenegative = 0
print("got here")
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

classifier = load_model('modelv4.h5')

import numpy as np
from keras.preprocessing import image
import os

print("Executing Human")

for files in os.listdir(r'C:\Users\x97272\newdata_set\validation_set\human'):
    pic = "C:\\Users\\x97272\\newdata_set\\validation_set\\human\\"+files
    test_image = image.load_img(pic, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result= classifier.predict(test_image)
    total += 1
    if result[0][0] == 1.0:
        totaltrupos += 1
        humantruepositive += 1
        leopardtruenegative += 1
        neithertruenegative += 1
    elif result[0][1] == 1.0:
        totalfalsepos += 1
        humanfalsenegative += 1
        leopardfalsepositive += 1
        neithertruenegative += 1
    else:
        totalfalsepos += 1
        humanfalsenegative += 1
        neitherfalsepositive += 1
        leopardtruenegative += 1
        
print("Human Complete")

print("Executing Leopard")

for files in os.listdir(r'C:\Users\x97272\newdata_set\validation_set\leopard'):
    pic = "C:\\Users\\x97272\\newdata_set\\validation_set\\leopard\\"+files
    test_image = image.load_img(pic, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result= classifier.predict(test_image)
    total += 1
    if result[0][0] == 1.0:
        totalfalsepos += 1
        humanfalsepositive += 1
        leopardfalsenegative += 1
        neithertruenegative += 1
    elif result[0][1] == 1.0:
        totaltrupos += 1
        leopardtruepositive += 1
        humantruenegative += 1
        neithertruenegative += 1
    else:
        totalfalsepos += 1
        neitherfalsepositive += 1
        leopardfalsenegative += 1
        humantruenegative += 1
        
print("Leopard Complete")

print("Executing Neither")

for files in os.listdir(r'C:\Users\x97272\newdata_set\validation_set\nothing'):
    pic = "C:\\Users\\x97272\\newdata_set\\validation_set\\nothing\\"+files
    test_image = image.load_img(pic, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result= classifier.predict(test_image)
    total += 1
    if result[0][0] == 1.0:
        totalfalsepos += 1
        neitherfalsenegative += 1
        humanfalsepositive += 1
        leopardtruenegative += 1
    elif result[0][1] == 1.0:
        totalfalsepos += 1
        neitherfalsenegative += 1
        leopardfalsepositive += 1
        humantruenegative += 1
    else:
        totaltrupos += 1
        neithertruepositive += 1
        humantruenegative += 1
        leopardtruenegative += 1
        
print("Neither Complete")

print("total pics: " + str(total))
print("humantruepositive: "+ str(humantruepositive))
print("humanfalsepositive: "+ str(humanfalsepositive))
print("humanfalsenegative: "+ str(humanfalsenegative))

print("leopardtruepositive: "+ str(leopardtruepositive))
print("leopardfalsepositive: "+ str(leopardfalsepositive))
print("leopardfalsenegative: "+ str(leopardfalsenegative))

print("neithertruepositive: "+ str(neithertruepositive))
print("neitherfalsepositive: "+ str(neitherfalsepositive))
print("neitherfalsenegative: "+ str(neitherfalsenegative))

truepositives = humantruepositive + leopardtruepositive + neithertruepositive
falsepositives = humanfalsepositive + leopardfalsepositive + neitherfalsepositive
truenegatives = humantruenegative + leopardtruenegative + neithertruenegative
falsenegatives = humanfalsenegative + leopardfalsenegative + neitherfalsenegative
#precision = (true positives)/(true positives + false positives)
print("Human")
precision = humantruepositive/(humantruepositive + humanfalsepositive)
print("Precision: " + str(precision))
#recall = true positive/(true positive + false negative)
recall = humantruepositive/(humantruepositive + humanfalsenegative)
print("Recall: " + str(recall))
#accuracy = (true positive+true negative)/total
accuracy = (humantruepositive + humantruenegative)/total
print("Accuracy: " + str(accuracy))

print("")
print("Leopard")
precision = leopardtruepositive/(leopardtruepositive + leopardfalsepositive)
print("Precision: " + str(precision))
#recall = true positive/(true positive + false negative)
recall = leopardtruepositive/(leopardtruepositive + leopardfalsenegative)
print("Recall: " + str(recall))
#accuracy = (true positive+true negative)/total
accuracy = (leopardtruepositive + leopardtruenegative)/total
print("Accuracy: " + str(accuracy))

print("")
print("Neither")
precision = neithertruepositive/(neithertruepositive + neitherfalsepositive)
print("Precision: " + str(precision))
#recall = true positive/(true positive + false negative)
recall = neithertruepositive/(neithertruepositive + neitherfalsenegative)
print("Recall: " + str(recall))
#accuracy = (true positive+true negative)/total
accuracy = (neithertruepositive + neithertruenegative)/total
print("Accuracy: " + str(accuracy))
    
#print("Human:0 Leopard:1 Neither:2")
