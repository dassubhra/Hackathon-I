import keyboard as kb
import Record_audio
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")


n_mfcc = 30
n_mels = 128
frames = 15

list_task = [['Idly', ' Dosa', ' Wada', 'Puri ', 'Chapathi'],[0, 1, 2, 3, 4]];
menu_prompts = ["menu_list.wav", "quantity_list.wav"]
#list_task[0] is list of menu items
#list_task[0] is list of quantity allowed

####################################################################################
#Train a classifier using the given training data to classify the samples
#One way is to regress the features to the get the labels
#get a confidence metric to evalaute your prediction
####################################################################################

T = 0.75 # confidence threshold, replace the value accordingly to your confidence measure

class order():
	
	#Prompts the menu to choose
	def prompt_menu(self, list_task, flag):#flag = menu(0)/quantity(1)
		Record_audio.play_audio(menu_prompts[flag])
		Record_audio.play_audio("beep.wav")
		for i in range(0,len(list_task[flag])):
			print('say '+ str(i) + ' for ' + str(list_task[flag][i]))

	#classify the input given the features and model. 
	#Change the function definition accordingly to suite your classifier 
	#It should return predicted label and a confident measure for your prediction
	def classify_input(self, features):
         data = []
         data.append(features)
         loadedModel = pickle.load(open("/Users/madhukaruturi/Desktop/Madhu/AI-ML/AIML-Labs/Hackathon I/model.sav",'rb'))
         predicted_label = int(loadedModel.predict(data)[0])
         confidence_label = 1
         return predicted_label,confidence_label

	def confirm_input(self, digit,confidence,flag):
		if(confidence > T) and (digit < len(list_task[flag])):
			return digit,list_task[flag][digit]
		else:
			print('Sorry we could not understand you, please reconfirm')
			#list_task = shuffled list_task
			Record_audio.play_audio("sorry_reconfirm.wav")
			return self.take_user_input(list_task,flag)


	def take_user_input(self, list_task,flag):
         self.prompt_menu(list_task,flag)
         audio_file_path = Record_audio.record_voice("userinput", "1", 1, "./")
         features = Record_audio.get_features(audio_file_path, sr=8000, n_mfcc=30, n_mels=128, frames = 15)
         digit,confidence = self.classify_input(features)
         digit,choice = self.confirm_input(digit,confidence,flag)
         return digit,choice

input_order = order()
flag = 0
digit1,choice1 = input_order.take_user_input(list_task, flag)
flag = 1
digit2,choice2 = input_order.take_user_input(list_task, flag)
print('you said digit ' + str(digit2) + ' and ' + str(digit1) + ' to confirm your order of ' + str(choice2) + ' quantity of ' + (choice1))
