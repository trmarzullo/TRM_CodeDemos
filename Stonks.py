import json
from datetime import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import pandas as pd
import ax
from ax.service.managed_loop import optimize
import time
import torch.utils.benchmark as benchmark



#import finnhub
#client = finnhub.Client(api_key="ca24922ad3iaqnc2nmlg")
#candles_data = client.stock_candles("AMD", "1", 1621317538, 1652839209)

#VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

def SplitUnixTime(unixtime):

    try:

        iter(unixtime)

    except:

        date_time = datetime.fromtimestamp(unixtime)

        unixday             = math.floor(unixtime / 86400)
        seconds_of_the_day  = (unixtime - (unixday * 86400) - 28800) / 57540

        day_of_week         = float(date_time.strftime("%w"))

        day_of_month        = float(date_time.strftime("%e"))

        return seconds_of_the_day, day_of_week, day_of_month

    else:

        seconds_of_the_day = np.zeros((len(unixtime), 1))
        day_of_week = np.zeros((len(unixtime), 1))
        day_of_month = np.zeros((len(unixtime), 1))

        for i in range(0, len(unixtime)-1):

            date_time = datetime.fromtimestamp(unixtime[i, 0])

            unixday                     = math.floor(unixtime[i, 0] / 86400)
            seconds_of_the_day[i, 0]    = (unixtime[i, 0] - (unixday * 86400) - 28800) / 57540

            day_of_week[i, 0]           = float(date_time.strftime("%w")) - 1

            day_of_month[i, 0]          = float(date_time.strftime("%e"))

        return seconds_of_the_day, day_of_week, day_of_month



#Candles Dataset

class CandleData(Dataset):

    def __init__(self, file_name, save_name, train_set = None):

        #Open and load data into Numpy array
        candles_json = json.load(open(file_name))
        candles_json.pop("s")

        k, v    = zip(*candles_json.items())
        candles = np.array(v).T

        #preview column for excel inspection
        real_prices = candles[:, 0:1]
        
        #assign "buy" and "no buy" labels
        labels = []

        for i in range(0, candles.shape[0]):

            buy = 0

            for price in candles[i+1:i+15, 0]:
                if price > candles[i, 0] * 1.002:
                    buy = 1

            labels.append(buy)

        #print("Buys")
        #print(labels.count(True))
        #print("No Buys")
        #print(labels.count(False))

        #Take date and split into feature vectors

        unixtimes = candles[:, 4:5]
        candles = np.delete(candles, 4, 1)
        dayseconds_column, weekday_column, monthday_column = SplitUnixTime(unixtimes)

        monthday_column = monthday_column / 31

        candles = np.hstack((candles, dayseconds_column, weekday_column, monthday_column))

        #Change absolute prices and volumes to relative prices and volumes 
        for i in range(candles.shape[0] - 1, 0 - 1, -1):

            if i == 0:
                candles[i, 0:5] = 0
            else: 
                candles[i, 0:5] -= candles[i-1, 0:5]
            
        #These Features have somewhat normal distributions
        #plt.hist(candles[:, 4], bins=30, range=(0, 20))
        #plt.show()
        
        #normalize volume
        if train_set == None:
            self.mean = np.mean(candles[:, 4:5])
            self.std = np.std(candles[:, 4:5])
            candles[:, 4:5] = (candles[:, 4:5] - self.mean) / self.std
            
        else:
            candles[:, 4:5] = (candles[:, 4:5] - train_set.mean) / train_set.std
       
        #one-hot the days of the week

        #JANK SHIT, BEWARE#####################
        candles_json = json.load(open("data/1 year AMD candles_1min.json"))
        candles_json.pop("s")
        k, v    = zip(*candles_json.items())
        tempcandles = np.array(v).T

        for i in range(0, tempcandles.shape[0]):
            time = datetime.fromtimestamp(tempcandles[i, 4])
            tempcandles[i, 4] = float(time.strftime("%w")) - 1 
        #############################################

        enc = preprocessing.OneHotEncoder(dtype=np.float32)
        enc.fit(tempcandles[:, 4:5])
        dayhotarray = enc.transform(candles[:, 6:7]).toarray()
        
        candles = np.hstack(((candles[:, :6], dayhotarray, candles[:, 7:8]))).astype(np.float32)
        
        #save to csv for inspection
        header = ["Real Price (meta)", "Closing Price", "High Price", "Low Price", "Open Price", "Volume", "Second of Day", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Day of Month", "Labels"]
        csv = pd.DataFrame(np.hstack((real_prices, candles, np.array([labels]).T)), columns = header)
        csv.to_csv(save_name, index=False, header=True)

        #save to object
        self.days = candles[:, -1]
        self.candles = torch.from_numpy(candles)
        #self.candles = self.candles.to(device)
        self.labels = labels

    def __getitem__(self, index):

        current_day = self.days[index]
        current_index = index - 1
        lens = []
        while (self.days[current_index] == current_day) \
        and (current_index >= 0) \
        and (index - current_index < 30):
            current_index -= 1
        
        return (self.candles[current_index+1:index+1, :] , self.labels[index])


    def __len__(self):

        return self.candles.shape[0]


#RNN Model

class RNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers):

        super(RNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.first_linear = nn.Linear(hidden_size, 20)
        self.final_linear = nn.Linear(20, output_size)

        self.leaky = torch.nn.LeakyReLU()

    def forward(self, x):

        #t0 = benchmark.Timer(
        #stmt='pack_sequence(x, enforce_sorted=False)',
        #setup='from torch.nn.utils.rnn import pack_sequence',
        #globals={'x': x})
        #print(f"PACK TIME {t0.timeit(100)}")
        
        out = self.lstm(x)
        out = out[1][0]

        out = self.leaky(self.first_linear(out))

        out = self.final_linear(out)

        #out = torch.sigmoid(self.final_linear(out))

        return out





def Sequence_Collate(batch):

    x, y = zip(*batch)

    labels = np.vstack(y)
    labels  = torch.from_numpy(labels).float()

    packed_x = torch.nn.utils.rnn.pack_sequence(x, enforce_sorted=False)

    return packed_x, labels

def Unzip(batch):

    x, y = zip(*batch)

    return x, y

def Train(model, dataloader, optimizer, duration):

    criterion = nn.BCEWithLogitsLoss()
    t = time.time()
    losses = []

    while time.time() - t < duration:


        loss = 0

        for batch_num, (x, y) in enumerate(dataloader):

            if time.time() - t < duration:

                labels = y.cuda(non_blocking = True)

                outputs = model.forward(x.cuda())
                outputs = outputs[0, :, :]

                loss = criterion(outputs, labels)
                
                optimizer.zero_grad(set_to_none=True)        
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        print(loss.item())
        
    return losses


def Test(model, dataloader):

    with torch.no_grad():
            incorrect_buys = 0
            incorrect_holds = 0
            correct_plays = 0
            total_count = 0

            for sample_num, (x, y) in enumerate(dataloader):

               
                    outputs = model.forward(x)
                    outputs = outputs[0, :, :]

                    labels = np.vstack(y)
                    labels  = torch.from_numpy(labels).float().to(device)

                    for i in range(outputs.shape[0]):

                        if outputs[i, 0] > 0.5 and y[i] != True:
                            incorrect_buys += 1
                        elif outputs[i, 0] < 0.5 and y[i] != False:
                            incorrect_holds += 1
                        else:
                            correct_plays += 1
                        total_count += 1

    return correct_plays, total_count

def Experiment(parameterization):

    print(f"HEYYYYYYYYYYYYYYYYYYYYYYYYYYY {parameterization}\n\n\n")
    trial_results = []

    for i in range(0, parameterization["num_repeats"]):
        ###Set up model and parameters
        batch_size = parameterization["batch_size"]
        data = CandleData("data/1 year AMD candles_1min.json", "Candle_Train_Preview.csv")
        dataloader = DataLoader(dataset = data, batch_size = batch_size, shuffle=True, collate_fn = Sequence_Collate, num_workers = 0, pin_memory=True)

        testdata = CandleData("data/recent2 AMD candles_1min.json", "Candle_Test_Preview.csv", data)
        testdataloader = DataLoader(dataset = testdata, batch_size = 32, shuffle=False, collate_fn = Sequence_Collate, num_workers = 0, pin_memory=True)

        input_size = data[0][0].shape[1]
        output_size = 1

        model = RNN(input_size, output_size, parameterization["hidden_size"], parameterization["num_layers"])
        model.to(device)

        optimizer = None

        if parameterization["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=parameterization["lr"])
        elif parameterization["optimizer"] == "ADAM":
            optimizer = torch.optim.Adam(model.parameters(), lr=parameterization["lr"])

        
        #Execute training
        for i in range(0, parameterization["num_repeats"]):
            losses = Train(model, dataloader, optimizer, parameterization["time_duration"])

            print(f"TRIAL {i} minimum loss: {min(losses)}")
            trial_results.append(min(losses))

    mean = 0
    sme  = 0
    if parameterization["num_repeats"] > 1:
        mean = np.mean(trial_results)
        sme  = np.std(trial_results) / len(trial_results)
    else:
        mean = trial_results[0]
        sme = 0
     
    print(f"FINAL MEAN LOSS FOR EXPERIMENT {mean}\n SME:{sme}")
    return (mean, sme)

    #y = range(1, len(losses)+1)
    #print(len(losses))
    #plt.plot(y, losses)
    #plt.show()

    #correct_plays, total_count = Test(model, testdataloader)
    #print(f"TOTAL CORRECT PLAYS: {correct_plays} / {total_count} \n ACCURACY {correct_plays / total_count * 100}%")

 


#MAIN LOOP#------------------------------------------------------------------------------------------------


def AxOptim():

    best_parameters, values, experiment, model = optimize(
        parameters=[
            {
                "name": "lr",
                "type": "range",
                "bounds": [0.01, 0.05],
                "value_type": "float",  # Optional, defaults to inference from type of "bounds".
                "log_scale": False,  # Optional, defaults to False.
            },
            {
                "name": "hidden_size",
                "type": "range",
                "bounds": [500, 1000],
                "value_type": "int"
            
            },
            {
                "name": "num_layers",
                "type": "range",
                "bounds": [5, 10],
                "value_type": "int"
            },
            {
                "name": "time_duration",
                "type": "fixed",
                "value": 1800,
                "value_type": "int",  # Optional, defaults to inference from type of "bounds".
            },
            {
                "name": "num_repeats",
                "type": "fixed",
                "value": 1,
                "value_type": "int",  # Optional, defaults to inference from type of "bounds".
            },
            {
                "name": "batch_size",
                "type": "fixed",
                "value": 64,
                "value_type": "int",  # Optional, defaults to inference from type of "bounds".
            },
            {
                "name": "optimizer",
                "type": "choice",
                "values": ["SGD", "ADAM"],
                "value_type": "str",  # Optional, defaults to inference from type of "bounds".
            }
        ],
        experiment_name= "Stonks",
        objective_name= "Loss",
        evaluation_function= Experiment,
        minimize= True,  # Optional, defaults to False.
        #parameter_constraints=["x1 + x2 <= 20"],  # Optional.
        #outcome_constraints=["l2norm <= 1.25"],  # Optional.
        total_trials=8, # Optional.
    )

    print(best_parameters)
    print(values)
    print(experiment)
    print(model)


if __name__ == '__main__':

    if torch.cuda.is_available():
        print("CUDA is available")
        device = "cuda"
    else:
        print("WARNING CUDA is not available WARNING")
        device = "cpu"

    parameterization = {
        "lr": 0.01,
        "hidden_size": 600,
        "num_layers": 7,
        "time_duration": 5000,
        "num_repeats": 1,
        "batch_size": 64,
        "optimizer": "SGD"
    }

    time_duration = 500

    #AxOptim()
    Experiment(parameterization)


    #save/load logic for reference
    #if epoch != 0 and epoch % 50 == 0:
    #    saves = saves + 1
    #    torch.save(model.state_dict(), "babys_first_model_" + str(saves) + ".pt")
    #    print("MODEL SAVED")
    #model.load_state_dict(torch.load("babys_first_model_7.pt"))