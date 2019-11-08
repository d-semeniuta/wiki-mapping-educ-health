import util_data
import pickle
import baseline.baselines

with open('../baseline/dumps/ed_baselines.pickle', 'rb+') as f:
    ed_baselines = pickle.load(f)

with open('../baseline/dumps/health_baselines.pickle', 'rb+') as f:
    health_baselines = pickle.load(f)

# hp tuning with Rwanda and Angola

# train in
#   Kenya  (2011-2015): nmothers = 23102, ndeaths= 502,  nbirths=15133,  IMR=0.0331725368400185,   under-ed=0.6944420396502466, over-ed=0.30555796034975324
#   Malawi (2011-2015): nmothers = 18977, ndeaths= 669,  nbirths=16999,  IMR=0.03935525619154068,  under-ed=0.7705116720240291, over-ed=0.2294883279759709
#   Myanmar (2011-2015): nmothers= 7795,  ndeaths=197,   nbirths=4848,   IMR=0.040635313531353134, under-ed=0.6379730596536242, over-ed=0.3620269403463759
#       wealthy
#   Ethiopia (2011-2015): nmothers=9961,  ndeaths=494,   nbirths=10230,  IMR=0.048289345063538616, under-ed=0.8655757454070876,  over-ed=0.13442425459291235

# test in
#   Uganda (near Kenya and Malawi) (2011-2015): nmothers=13527,ndeaths=625, nbirths=15242,IMR=0.04100511743865634, under-ed=0.7451023878169587, over-ed=0.2548976121830413
#   Philippines (far from all)     (2011-2015): nmothers=15308, ndeaths=237, nbirths=10650, IMR=0.022253521126760562, under-ed=0.2187744969950353, over-ed=0.7812255030049647
#   Mozambique (2011-2015): nmothers=4134, ndeaths=118.99999999999999, nbirths=4861, IMR=0.024480559555646982, under-ed=0.7561683599419449, over-ed=0.2438316400580552


#   Angola (2011-2015): nmothers = 11148,,ndeaths= 566,  nbirths=14170,  IMR=0.03994354269583627, under-ed=0.6881054897739505, over-ed=0.31189451022604947
#   India  (2011-2015): nmothers = 445091, ndeaths=9501, nbirths=226507, IMR=0.04194572353172308, under-ed=0.5085072490794018, over-ed=0.4914927509205983


placeholder = 1