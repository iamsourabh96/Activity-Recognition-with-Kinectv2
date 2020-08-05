import numpy as np
import glob
import os
import numpy as np

class DataLoader():
    
    def getData(self):
        activity = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[],18:[]}
        train = 0
        for file in glob.glob("./Data/data/*txt"):
            train+=1
            action_np = np.array([0,0,0]).reshape(-1,3)
            filename = os.path.basename(file)
            label = int(filename[1:3])    
        
            with open(file,"r") as f:
                skeleton_data =f.readlines()    
                # frames = len(skeleton_data)//15    
                action = []
                body =[]
                joints = 0
                for joint in skeleton_data:
                    joints+=1
                    joint = joint.replace("\n", "")
                    joint =  np.fromstring(joint, dtype=np.float, sep=" ")
                    body.append(joint)
                    
                    if not joints%15:

                        action.append(body)
                        action_np = np.vstack([action_np,np.array(body).reshape(-1,3)])
                        if train <= 500:
                            np.save("./Data/train/"+filename[:-4]+".npy",action_np[1:])
                        else:
                            np.save("./Data/test/"+filename[:-4]+".npy",action_np[1:])
                        body=[]
                activity[label].append(action)
        return activity
    

    def preprocess(self,array,size=40):
        length = array.shape[0]
        remove = length//15 - size
        if not remove%2:
            remove = (int(remove*15/2))
            return array[remove:length-remove]
        else:
            remove+=1
            remove = (int(remove*15/2))
            return array[remove-15:length-remove]
        
    def oneHotEncoding(self,idx):
        vec = np.zeros((18,1))
        vec[idx] = 1
        return vec
    
    
    def norm(self,data):
        if abs(min(data)) > max(data):
            max_val = abs(min(data))
            min_val = min(data)
        else:
            max_val = max(data)
            min_val = -max(data)
        norm_val = (2*data)/(max_val - min_val)
        return norm_val
        
    
    def normalize(self,array):        
        array[:,0] = self.norm(array[:,0])
        array[:,1] = self.norm(array[:,1])
        array[:,2] = self.norm(array[:,2])
        return array
            
        
        



            

        
        
        

        
        
        
        
    

