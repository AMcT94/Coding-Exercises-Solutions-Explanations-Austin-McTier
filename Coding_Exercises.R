
# Coding Exercises - Austin McTier

# Please create a public git repo in github/gitlab/bitbucket for your solutions and share the link. Please also include
# your code, solutions, and any README, solution explanations etc. that you may have within this
# repo. You may use any programming language of your choice, but Python or R are preferred.


####### 1. Unsupervised + supervised learning

# Attached is a data file dataClustering.csv which contains a data set of 2500 samples with 8 features.

############ i. Perform any clustering of your choice to determine the optimal # of clusters in the data

# install and call any necessary packages

install.packages("factoextra")
install.packages("ggplot2")
install.packages("plyr")
install.packages("dplyr")
install.packages("clustree")
install.packages("cluster")
install.packages("NbClust")
install.packages("cowplot")
install.packages("clValid")
install.packages("knitr")
install.packages("kableExtra")
install.packages("VGAM")
install.packages('caret')
install.packages('randomForest')
install.packages('datasets')
install.packages("sgd")
install.packages("e1071")
install.packages("pracma")
install.packages("kalmanfilter")
install.packages("forecast")
install.packages("plyr")
install.packages("tidyr")
install.packages("dplyr")
install.packages("corrplot")
install.packages("Hmisc")
install.packages("sgd")

library(factoextra)
library(ggplot2)
library(plyr)
library(dplyr)
library(clustree)
library(cluster)
library(NbClust)
library(cowplot)
library(clValid)
library(knitr)
library(kableExtra)
library(VGAM)
library(e1071)
library(caret)
library(randomForest)
library(datasets)
library(nnet)
library(pracma)
library(kalmanfilter)
library(forecast)
library(plyr)
library(tidyr)
library(dplyr)
library(corrplot)
library(Hmisc)
library(sgd)

# set working directory for where data is located
user <- "C:/Users/f1abm01/Work Folders/Documents/Job_Questions/"
user <- "C:/Users/amcti/OneDrive/Documents/Job_Questions/"

setwd(user)

# load the data into the environment 

data <- read.csv(paste0(user,"dataClustering.csv"), header=FALSE)

# I am personally more familiar with K-means clustering than other clustering algorithms, so I will use that.
# Will look at a number of different methods to determine the optimal number of clusters. First, I will want to visualize
# the clusters for different k values to get an idea of what could be the optimal number of clusters 


kmean_calc <- function(df, ...){
  kmeans(df, scaled = ..., nstart=30)
}
km2 <- kmean_calc(data, 2)
km3 <- kmean_calc(data, 3)
km4 <- kmeans(data, 4)
km5 <- kmeans(data, 5)
km6 <- kmeans(data, 6)
km7 <- kmeans(data, 7)
km8 <- kmeans(data, 8)
km9 <- kmeans(data, 9)
km10 <- kmeans(data, 10)
km11 <- kmeans(data, 11)
p1 <- fviz_cluster(km2, data = data, frame.type = "convex") + theme_minimal() + ggtitle("k = 2") 
p2 <- fviz_cluster(km3, data = data, frame.type = "convex") + theme_minimal() + ggtitle("k = 3")
p3 <- fviz_cluster(km4, data = data, frame.type = "convex") + theme_minimal() + ggtitle("k = 4")
p4 <- fviz_cluster(km5, data = data, frame.type = "convex") + theme_minimal() + ggtitle("k = 5")
p5 <- fviz_cluster(km6, data = data, frame.type = "convex") + theme_minimal() + ggtitle("k = 6")
p6 <- fviz_cluster(km7, data = data, frame.type = "convex") + theme_minimal() + ggtitle("k = 7")
plot_grid(p1, p2, p3, p4, p5, p6, labels = c("k2", "k3", "k4", "k5", "k6", "k7"))

# Now I will go through a couple of different methods to see what the optimal number of clusters could be 

# A. The Elbow Method - sum of squares at each number of clusters is calculated and graphed
fviz_nbclust(data, kmeans, method="wss", k.max=24) + theme_minimal() + ggtitle("the Elbow Method")

# B. The Silhouette Method - computes the average silhouette of observations for different values of k. The optimal number of clusters k is the one that maximize the average silhouette over a range of possible values for k.
fviz_nbclust(data, kmeans, method = "silhouette", k.max = 24) + theme_minimal() + ggtitle("The Silhouette Plot")

# C. The Gap Statistic 
gap_stat <- clusGap(data, FUN = kmeans, K.max = 24, B = 50)
fviz_gap_stat(gap_stat) + theme_minimal() + ggtitle("fviz_gap_stat: Gap Statistic")

# D. NbClust - provides 30 indices for determining the relevant number of clusters and proposes to users the best clustering scheme from the different results obtained by varying all combinations of number of clusters, distance measures, and clustering methods.

res.nbclust <- NbClust(data, distance = "euclidean",
                       min.nc = 2, max.nc = 9, 
                       method = "complete", index ="all")

# Based on the results, it seems that the optimal # of clusters is either 3 or 4. I am going to go w/ 3 given the Gap Plot & Silhouette Plot results. 
# I do, however, understanding that there are other clustering methods (i.e. Hierarchical, PAM, etc.) that may be better than K-means clustering in this case.
# So, as an aside, I will check to see if I'm using the optimal clustering method

valid <- clValid(data, nClust = 2:24, maxitems = 2500,
                  clMethods = c("hierarchical","kmeans","pam"), validation = "internal")
# Summary
summary(valid) %>% kable() %>% kable_styling()

# According to the results, hierarchical clustering would be the preferable clustering method, but 3-4 clusters still appears to be the optimal number of clusters
# For the purpose of ii, I will be using 3 clusters 


############ ii. Using the result of i) assign clusters labels to each sample, so each sample's label is the
#                cluster to which it belongs. Using these labels as the exact labels, you now have a labeled dataset.
#                Build a classification model that classifies a sample with its corresponding label. Use multinomial 
#                regression as a benchmark model, and any ML model (trees, forests, SVM, NN etc.) as a comparison model.
#                Comment on which does better and why.


# Given that I used k-means clustering to determine the optimal number of clusters, I will now assign cluster labels to each sample

k <- kmeans(data, 3) 
names(k)
data <- data.frame(data, K=k$cluster)

# so now "K" in our dataset is the cluster label, assigning a cluster label to each data point. 

# We now want to build a classification model that classifies a sample with its corresponding label. We will use a multinomial regression as a benchmark model.
# For a comparison model, I will be using a randomForest model 

# Convert the cluster variable to a factor
data$K <- as.factor(data$K)

# separate the data into training data (70 % ) and test data (30 % ), to see which model is a better means of classification.
dt = sort(sample(nrow(data), nrow(data)*.7))
train<-data[dt,]
test<-data[-dt,]

# Fit a multinomial logistic regression model
multi <- multinom(K ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8,
            data = train)

# Show results for multinomial regression model
summary(multi)

# Fit a random forest model
rf <- randomForest(K ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8,
            data = train, proximity=TRUE)

# Show results for random Forest model
print(rf)

# predict both models on the test data
pred_mn <- predict(multi, test)
pred_rf <- predict(rf, test)

# Use a confusion matrix to determine how well the models classify new data
confusionMatrix(pred_rf, test$K)
confusionMatrix(pred_mn, test$K)

# For some reason, both models perfectly classify the test data without any misclassification. So both models perform exactly the same 
# I am unsure if that was suppose to be the result or if I messed something up in my process. I also tried doing so on the entire data, 
# as well as trying 4 clusters instead of 3, and got the same result. 


#------------------------------------------------------------------------------------------------------------------------------------------------------------


####### 2. Prediction + filtering

# Attached are 3 files: xvalsSine.csv, cleanSine.csv and noisySine.csv. xvalsSine.csv contains
# 1000 x-values in the interval -pi/2 to pi/2. cleanSine.csv is a pure sine(x) function for the
# x values mentioned earlier. noisySine.csv contains sine(x) corrupted by noise. 

############ i. Using xvalsSine.csv and cleanSine.csv as a labeled dataset (x,sine(x)) being (value,label) with a
#               random train/test split of 0.7/0.3, build an OLS regression model (you may want to use polynomial basis of a sufficiently large order). 

#rm(list=ls())
#user <- "C:/Users/f1abm01/Work Folders/Documents/Job_Questions/"
#setwd(user)

# load the data into the environment 
cleanSine <- read.csv(paste0(user,"cleanSine.csv"), header=FALSE)
xvalsSine <- read.csv(paste0(user,"xvalsSine.csv"), header=FALSE)

data <- as.data.frame(cbind(xvalsSine, cleanSine))
colnames(data) <- c("x", "sine(x)")

# Separate the data into training data (70%) and test data (30%)
dt = sort(sample(nrow(data), nrow(data)*.7))
train<-data[dt,]
test<-data[-dt,]

# OLS regression model 
model <- lm(`sine(x)` ~ poly(x,19), train)
summary(model)

# Predicts the values with confidence interval
pred <- predict(model, test)

# Root mean square error to determine performance 
rmse <- sqrt(mean((test$`sine(x)` - pred)^2))
rmse 

# Normalized RMSE = RMSE / std dev
nrmse <- rmse/sd(pred)
nrmse

########### (bonus) If you used the normal equations to solve the OLS problem, can you redo it with stochastic gradient descent? (What do you mean by "normal"?)
sgd.theta <- sgd(`sine(x)` ~ x, train, model="lm")
predict(sgd.theta, test, type="link")


# For some reason, the 'predict' function doesn't work w/ sgd, due to something wrong w/ how it's formatted, and how it doesn't play nice w/ predict, resulting in the
# Matrix multiplication not work. Returns error: "Error in newdata %*% coef(object) requires numeric/complex matrix/vector arguments" .

# Attemp at predict.sgd, still not working 
predict.sgd <- function(object, newdata, type="link", ...) {
  if (!(object$model %in% c("lm", "glm", "m"))) {
    stop("'model' not supported")
  }
  if (!(type %in% c("link", "response", "term"))) {
    stop("'type' not recognized")
  }
  
  if (object$model %in% c("lm", "glm")) {
    if (type %in% c("link", "response")) {
      eta <- newdata %*% coef(object)     ## This is where the error occurs
      if (type == "response") {
        y <- object$model.out$family$linkinv(eta)
        return(y)
      }
      return(eta)
    }
    eta <- newdata %*% diag(coef(object)) ## This is where the error occurs
    return(eta)
  } else if (object$model == "m") {
    if (type %in% c("link", "response")) {
      eta <- newdata %*% coef(object)
      if (type == "response") {
        y <- eta
        return(y)
      }
      return(eta)
    }
    eta <- newdata %*% diag(coef(object))
    return(eta)
  }
}

# Would try and do manually if allotted enough time, tried my best down below

# beta = (XtX)-1Xty
# y = beta * X
y <- sgd.theta$coefficients[1] + sgd.theta$coefficients[2]*test[,1] 

# RMSE
rmse <- sqrt(mean((test$`sine(x)` - y)^2))
print(rmse)

nrmse <- rmse/sd(pred)
nrmse

########### ii. Now, assume you are given the noisySine.csv as a time series with the values of  xvalsSine.csv being the time variable. Filter the noisySine.csv data with any filter of your choice and compare against cleanSine.csv to report the error.

# create data set for noisy sine data 
noisySine <- read.csv(paste0(user,"noisySine.csv"), header=FALSE)
dataNoisy <- as.data.frame(cbind(xvalsSine, noisySine))
colnames(dataNoisy) <- c("x", "sine(x)")

# Plot the clean sine data and the noisy sine data to get an idea of how noisy the latter is 
plot(x = dataNoisy$x, y=dataNoisy$`sine(x)`, type = "l", lwd=2, col = "blue", xlab="x", ylab="signal")
lines(x = dataNoisy$x, y = data$`sine(x)`, lwd=2, col = "yellow")
 
# Estimate the signal-to-noise to get a better idea of how noisy the data is relative to it's clean version. 
s_to_n <- max(data$`sine(x)`)/sd(dataNoisy$`sine(x)`)
s_to_n # = 1.401357


# Will use R's filter() function to smooth noise and remove background signals. Could use R's fft() function for Fourier filtering, but I am personally not as familiar with that one 
# Using a 50-point moving average for the filter

mov_avg_e = rep(1/50, 50)
noisy_signal_movavg <- stats::filter(dataNoisy$`sine(x)`, mov_avg_e)
plot(x = data$x, y = noisy_signal_movavg, type = "l", lwd = 2, col = "blue", xlab = "x", ylab = "signal")
lines(x = data$x, y = data$`sine(x)`, lwd = , col = "red")

s_to_n_movavg <- max(data$`sine(x)`)/sd(noisy_signal_movavg, na.rm = TRUE)
s_to_n_movavg # = 1.385845

# average error 
mean(data$`sine(x)` - noisy_signal_movavg, na.rm=T) # = -0.00259983
# root mean squared error 
sqrt(mean((data$`sine(x)` - noisy_signal_movavg)^2, na.rm=T)) # = 0.01551806


########### (bonus) Can you code a Kalman filter to predict out 10 samples from the noisySine.csv data?
# Tried the OLS model I did for i) but it gave an error, "Error in KalmanForecast(n.ahead = 10, fit$model) : invalid argument type", so trying an auto.arima instead 

fit <- auto.arima(dataNoisy$`sine(x)`)
fit
KalmanForecast(n.ahead = 10, fit$model)







# -------------------------------------------------------------------------------------------------------------------------------------------------------

####### 3. Time series with pi

# Attached is a function genPiAppxDigits(numdigits,appxAcc) which returns an approximate value of pi
# to numdigits digits of accuracy. appxAcc is an integer that controls the approximation accuracy, with
# a larger number for appxAcc leading to a better approximation.


################ i) Fix numdigits and appxAcc to be sufficiently large, say 1000 and 100000 respectively.
#                   Treat each of the 1000 resulting digits of pi as the value of a time series. Thus x[n]=nth digit
#                   of pi for n=1,1000. Build a simple time series forecasting model (any model of your choice)
#                   that predicts the next 50 digits of pi. Report your accuracy. Using your results, can
#                   you conclude that pi is irrational? If so, how?


# I was unable to find / produce an adequate equivalent for the function below in R, since R seems to limit the possible number of digits for pi to 1000 digits. 
# Therefore, I use python 3.11 to use the functions and generate the value of pi up to the Nth digit, while using R to do the time series analysis. 

# Clear environment
rm(list=ls())

# assign pi value w/ 1000 digits and with appxAcc value of 100000
genpi_1000_100000 <- "3.141582653589793488462643352029502893728419393964949575908635991959269055172987594527216824734804529137861641083337440336723053487259890992088747229299047399846134629044413543139416286826959021166889889521872086543106423174088208073273528179653154867114850775005630652689355017995935043277913907341820729125222065618237414094343449118572789088878914761256733164157367924367201963360271333371830777757267030650568358361430699274713913748677596264883595033338551714160625466439196638710307218241361002027498920675667065328760607271980066818068962250363986500332016550745383124476109952309914657918295498358072114118703497939467135749953688983232621258472729625916154308720391124366442889880343097291695390186908964498564653971997671852967834502770272641394138236345959387145635490724473079727986491778576447668983542943927159874629275044567471514395067158063331483653009812460513442011542878120445659118725651537233770209932268364038249637616908488221806022742858419862979966848784298020632704125898238"

# form data so that it is in an easily readable time series format 
data <- as.data.frame(cbind(1, genpi_1000_100000))
data <- data %>% mutate(genpi_1000_100000 = strsplit(as.character(genpi_1000_100000),"")) %>%
  unnest(genpi_1000_100000)
data <- data[data$genpi_1000_100000 != ".",]
data$genpi_1000_100000 <- as.numeric(data$genpi_1000_100000)
genpi_1000_100000 <- ts(data$genpi_1000_100000)

# Trying out a seasonal naive fit model first, then trying out an auto.arima w/ best choices for p,d, and q.
fit <- snaive(genpi_1000_100000, h=50)
fit
next50 <- c(6,9,6,8,1,7,3,5,3,0,9,7,3,6,7,9,2,9,8,8,5,4,6,4,1,4,7,4,7,6,0,2,0,2,1,0,7,8,6,8,9,3,6,1,6,9,9,9,3,3)   # appxAcc = 1000

# Analyze accuracy of the seasonal naive fit model
accuracy(fit, next50)

# Trying auto arima function to see what best ARIMA model would work for this time series set
fit <- auto.arima(genpi_1000_100000)
fit
pred <- predict(fit, 50)

ME <- mean(next50 - pred$pred)
RMSE <- sqrt(mean((next50 - pred$pred)^2))

# I believe I can conclude that pi is irrational, though I'm uncertain about the strength of my proof. In the seasonal naive forecast I did, the autocorrelation of errors lag 1 (ACF1) is NA, indicating there is no time-based (serial)
# correlation amongst the digits of Pi as 'time' progresses. In addition, the ideal arima model for this time series analysis is ARIMA(0,0,0) meaning the optimal ARIMA model
# is one with no AR component, no I component, and no MA component, just a flat static forecast of one value into the future. 
# Both models performed terribly on the test data (the next 50 digits of pi)



#  (bonus) Now let's vary appxAcc to be 1000,5000,10000,50000,100000 with fixed numdigits=1000. You thus
#          have 5 time series, each corresponding to a value of appxAcc. Can you find the pairwise correlation
#          between each of the time series?

# Generating time series data for 1000 digits of pi based on approximation accuracies of 1000, 5000, 10000, 50000, and 100000, respectively. 
genpi_1000_100000 <- "3.141582653589793488462643352029502893728419393964949575908635991959269055172987594527216824734804529137861641083337440336723053487259890992088747229299047399846134629044413543139416286826959021166889889521872086543106423174088208073273528179653154867114850775005630652689355017995935043277913907341820729125222065618237414094343449118572789088878914761256733164157367924367201963360271333371830777757267030650568358361430699274713913748677596264883595033338551714160625466439196638710307218241361002027498920675667065328760607271980066818068962250363986500332016550745383124476109952309914657918295498358072114118703497939467135749953688983232621258472729625916154308720391124366442889880343097291695390186908964498564653971997671852967834502770272641394138236345959387145635490724473079727986491778576447668983542943927159874629275044567471514395067158063331483653009812460513442011542878120445659118725651537233770209932268364038249637616908488221806022742858419862979966848784298020632704125898238"
data <- as.data.frame(cbind(1, genpi_1000_100000))
data <- data %>% mutate(genpi_1000_100000 = strsplit(as.character(genpi_1000_100000),"")) %>%
  unnest(genpi_1000_100000)
data <- data[data$genpi_1000_100000 != ".",]
data$genpi_1000_100000 <- as.numeric(data$genpi_1000_100000)
genpi_1000_100000 <- ts(data$genpi_1000_100000)

genpi_1000_1000 <- "3.140592653839792925963596502869395970451389330779724489367457783541907931239747608265172332007670207231403885276038710899938066629552214564551237742887150050440512339302537072825852760246628025562008569471700451065826106184744099667808080815231833582150382088582680381403109153574884416966097481526954707518119416184546424446286573712097944309435229550466609113881892172898692240992052089578302460852737674933105951137782047028552762288434104643076549100475536363928011329215789260496788581009721784276311248084584199773204673225752150684898958557383759585526225507807731149851003571219339536433193219280858501643712664329591936448794359666472018649604860641722241707730107406546936464362178479780167090703126423645364670050100083168338273868059379722964105943903324595829044270168232219388683725629678859726914882606728649659763620568632099776069203461323565260334137877031715969991517031530618215153370441064935913039433435501768006700003562571354454019577851729026491381185793546112523014568827104"
data <- as.data.frame(cbind(1, genpi_1000_1000))
data <- data %>% mutate(genpi_1000_1000 = strsplit(as.character(genpi_1000_1000),"")) %>%
  unnest(genpi_1000_1000)
data <- data[data$genpi_1000_1000 != ".",]
data$genpi_1000_1000 <- as.numeric(data$genpi_1000_1000)
genpi_1000_1000 <- ts(data$genpi_1000_1000)

genpi_1000_5000 <- "3.141392653591793238362643395479500114198179818834553219696518762545891600633419497962998924770673168710283823839836179157453636766305824737524833968175020953999518276228928276135050836031522684352070141764416660950080705735709509049211009022030905888696882351973175109143721777156281885098401010615734271912451697118830805101893454605229723159327829661714529253161467880449107865768671570406459235535897806256027586486852701552669671213164728554998489176056638611925369118846903972488735108144725156176522263258592689341296066864527481176161830632359858264091924884401179484921417946154217814843488302389138078713124450057174871971699533127152324739783450353109886038572954524390618765688863828148491507841422212718558460029531829113547222750953497457453498984825771035314966867253863334332767498806594100460569270413983238412448829157890254440428324451765996114827160912165970428543723124740368488047985381125563216444090694430009774831632562794875121443053781017254689849476826442134414195182525038"
data <- as.data.frame(cbind(1, genpi_1000_5000))
data <- data %>% mutate(genpi_1000_5000 = strsplit(as.character(genpi_1000_5000),"")) %>%
  unnest(genpi_1000_5000)
data <- data[data$genpi_1000_5000 != ".",]
data$genpi_1000_5000 <- as.numeric(data$genpi_1000_5000)
genpi_1000_5000 <- ts(data$genpi_1000_5000)

genpi_1000_10000 <- "3.141492653590043238459518383374815378787013642744180460513479805474395670690028850870632943186765515712449180270879595216656138346723053240857425165370145476523667024194024852565525340949987938859407291090084369607533590093011356667536474383196232294416124465212973402737560218902832545598911114921763309123494546725137639794421211259780769270774350166166629350196721118541891276060866823619456651919503428074974421806424971679589995863120976588279052542717840680776462585504782281906759319859180054540823179806525272024045902037633147888653145501081203177123171687280560128582629967290995675661691157590271562777714457488328729523366397054559517627774722994296656826379543178019711693943041940888313350656693540172051738993799856726199543222929370419651682911520325983331999806599285846266894484289814047651166382620236905919106916891530214778428395564534380609826459502020658951813302998614022419858921197044547445877341616578057960931567875459021314860786833928737954470235475638065900486208612260"
data <- as.data.frame(cbind(1, genpi_1000_10000))
data <- data %>% mutate(genpi_1000_10000 = strsplit(as.character(genpi_1000_10000),"")) %>%
  unnest(genpi_1000_10000)
data <- data[data$genpi_1000_10000 != ".",]
data$genpi_1000_10000 <- as.numeric(data$genpi_1000_10000)
genpi_1000_10000 <- ts(data$genpi_1000_10000)

genpi_1000_50000 <- "3.141572653589795238462642383279504104197166629375115925174890537008215128244330696247132418110194705376872654850187409650123878937885455768573397484782581400540506010392838515885612892534646194286367054557195683394732831249771167253883796744949804323836056449674327916273168149408831018437531958649861904599540569902601568763349765671796236679779205911787878977305395328808078712955284171264479845731528221119004141295361301019101515985767667204965746596162903056722108799673824865905411739957699428635767242430081816545156332932405088962280497168899286442736315643942747862383831069832267070165222476306447418096610525374334781052194024198592871565664301855487217790284749236413786887090078815469316287738031969901109864792181048575494727235999869054992492406022453442675049555314736871404648477465505942972919970744578472974455178391623942300365904491653310044984363768165105098842193772316736706024279011674691845462398027617955326562465168235353516658876946772861754199243571344064138544946006643"
data <- as.data.frame(cbind(1, genpi_1000_50000))
data <- data %>% mutate(genpi_1000_50000 = strsplit(as.character(genpi_1000_50000),"")) %>%
  unnest(genpi_1000_50000)
data <- data[data$genpi_1000_50000 != ".",]
data$genpi_1000_50000 <- as.numeric(data$genpi_1000_50000)
genpi_1000_50000 <- ts(data$genpi_1000_50000)

#remove data to make environment a bit cleaner
rm(data)

# look at the pairwise correlation between all 5 time series
rcorr(cbind(genpi_1000_1000, genpi_1000_5000, genpi_1000_10000, genpi_1000_50000, genpi_1000_100000))

# Create correlation plot to visualize the correlation between all 5 time series 
mydata <- data.frame(cbind(genpi_1000_1000, genpi_1000_5000, genpi_1000_10000, genpi_1000_50000, genpi_1000_100000))
corrplot(cor(mydata))

#def genPiAppxDigits(numdigits,appxAcc):
#	import numpy as np
#	from decimal import getcontext, Decimal
#	getcontext().prec = numdigits
#	mypi = (Decimal(4) * sum(-Decimal(k%4 - 2) / k for k in range(1, 2*appxAcc+1, 2)))
#	return mypi











