# R script for creating prediction profiles of a network based on its weights
#
# for the paper "A neural network model for the evolution of learning in changing environments"					
# by Magdalena Kozielska and Franz J.Weissing

# (c) Magdalena Kozielska 2023



rm(list = ls())

##########################


# network - 1 input, 2 hidden layers each with 4 nodes, 1 output
# biases and weights
#	b2		b3		b4		b5		b6		b7		b8		b9		b10	
#  w[1]	w[2]	w[3]	w[4]	w[5]	w[6]	w[7]	w[8]	w[9]	
#	w12		w13		w14		w15		w26		w27		w28		w29		w36		w37		w38		w39		w46		w47		w48		w49		w56		w57		w58		w59		w610	w710	w810	w910
#	w[10]	w[11]	w[12]	w[13]	w[14]	w[15]	w[16]	w[17]	w[18]	w[19]	w[20]	w[21]	w[22]	w[23]	w[24]	w[25]	w[26]	w[27]	w[28]	w[29]	w[30]	w[31]	w[32]	w[33]

#######################################################


# clamped ReLU function
cReLU <- function(x){
  if(x>1) {return(1)}
  else if (x<0) {return(0)}
  else {return(x)}
}

# network with given weights m_w calculates output given input 
network_function <- function(m_w, input){
  N2 = 0.0 
  N3 = 0.0 
  N4 = 0.0
  N5 = 0.0
  N6 = 0.0
  N7 = 0.0
  N8 = 0.0
  N9 = 0.0
  output = 0.0
  
  N2 = cReLU(m_w[10] * input + m_w[1])
  N3 = cReLU(m_w[11] * input + m_w[2])
  N4 = cReLU(m_w[12] * input + m_w[3])
  N5 = cReLU(m_w[13] * input + m_w[4])
  
  N6 = cReLU(m_w[14] * N2 + m_w[18] * N3 + m_w[22] * N4 + m_w[26] * N5 + m_w[5])
  N7 = cReLU(m_w[15] * N2 + m_w[19] * N3 + m_w[23] * N4 + m_w[27] * N5 + m_w[6])
  N8 = cReLU(m_w[16] * N2 + m_w[20] * N3 + m_w[24] * N4 + m_w[28] * N5 + m_w[7])
  N9 = cReLU(m_w[17] * N2 + m_w[21] * N3 + m_w[25] * N4 + m_w[29] * N5 + m_w[8])
  
  output = m_w[30] * N6 + m_w[31] * N7 + m_w[32] * N8 + m_w[33] * N9 + m_w[9];
  
  return(output)
}

##################################
# function linking environmental cue to environmental quality - bell-shaped - wrapping is done later

env_function <- function(m, sd, env) {  
  
  return(exp(-0.5 * ((env - m) * (env - m)) / (sd * sd)))
}

#####################################################
# MAIN

# !!!!!!!!!!!!don't forget to set working directory
setwd("")

#ENVIRONMENT

env_sd <- 0.25   # "standard deviation" of the environmental quality distribution
env_change <- 0.4   # m
ch_rate <- 10   # 1/f - environment changes every ch_rate generations
lifespan <- 500
initLE <- 0   # initial number LE

gen <- 60000
ind <- 0   # random individual as individuals are ordered randomly

all_data <- data.frame(cues=double(), quality=double(), rypes=character(), replicate=double())


for(rep in c(0:19)){
  #loading simulation data
  sim_code <- paste0("FixedLEperiod_LS_",lifespan,"_initLE_",initLE,"_envSD_",env_sd,"_envChange_",env_change,"_envChangeRate_",ch_rate,"_random_", rep)
  dataW <- read.csv(file = paste0(sim_code, "_Weights.csv"), header = TRUE, sep = ",")
  dataA <- read.csv(file = paste0(sim_code, "_Averages.csv"), header = TRUE, sep = ",")
  dataD <- read.csv(file = paste0(sim_code, "_Details.csv"), header = TRUE, sep = ",")
  
  env_mean <- as.numeric(dataA$environmental.mean.location[dataA$Generation==gen])    # cue linked with the best environment

  ind_weights <- as.numeric(dataW[dataW$Generation==gen & dataW$Individual == ind,c(4:36)])
  ind_weights_after <- as.numeric(dataW[dataW$Generation==gen & dataW$Individual == ind,c(4:32,37:40)])
  
  
  cues <- seq(from = -1.0, to = 1.0, by = 0.05)
  exp_quality_before <- c()  # expected quality before learning
  exp_quality_after <- c()  # expected quality after learning
  real_quality <- c()   # real environmental quality
  
  # calculate quality for each cue
  for(c in cues){
    exp_b <- network_function(ind_weights, c)  # before learning
    exp_a <- network_function(ind_weights_after, c)  # after learning
    
    exp_quality_before <- c(exp_quality_before, exp_b)
    exp_quality_after <- c(exp_quality_after, exp_a)
    
    # real quality 
    if (c < env_mean - 1) {
      real_q <- env_function(env_mean, env_sd, 2*1+c)
    }
    else if (c > env_mean + 1) {
      real_q <- env_function(env_mean, env_sd, c-2*1)
    }
    else {
      real_q <- env_function(env_mean, env_sd, c)
    }
    
    real_quality <- c(real_quality, real_q)
    
    
  }
  
  exp_b_df <- data.frame(cues, exp_quality_before)  
  exp_b_df$type <- "before"
  names(exp_b_df)[2]<-paste("quality")
  exp_a_df <- data.frame(cues, exp_quality_after)  
  exp_a_df$type <- "after"
  names(exp_a_df)[2]<-paste("quality")
  real_q_df <- data.frame(cues, real_quality)  
  real_q_df$type <- "real"
  names(real_q_df)[2]<-paste("quality")
  
  rep_data <- rbind(real_q_df, exp_b_df, exp_a_df)
  
  rep_data$replicate <- rep
  rep_data$LE <- as.numeric(dataD$Learning_episodes[(dataD$Generation==gen & dataD$Individual == ind)])
  rep_data$LR <- as.numeric(dataD$Learning_rate[(dataD$Generation==gen & dataD$Individual == ind)])
  
  # re-scaling between 0 and 1 for easier visualization
  
  before <- rep_data$quality[rep_data$type == "before"]
  after <- rep_data$quality[rep_data$type == "after"]
  rep_data$rescaled <- rep_data$quality
  rep_data$rescaled[rep_data$type == "before"] <- (rep_data$quality[rep_data$type == "before"] - min(before))/(max(before)-min(before))
  rep_data$rescaled[rep_data$type == "after"] <- (rep_data$quality[rep_data$type == "after"] - min(after))/(max(after)-min(after))

  all_data <- rbind(all_data, rep_data)
}

# save data
write.csv(all_data, paste0("N_1000_G_",gen,"_ind_",ind,"_initLE_",initLE,"_exp_quality_LS_",lifespan,"_envSD_",env_sd,"_envChange_",env_change,"_envChangeRate_",ch_rate,"_random.csv"), row.names = FALSE)

# load data for plotting

env_sd <- 0.25   # "standard deviation" of the environmental quality distribution
env_change <- 0.4
ch_rate <- 10
lifespan <- 500
initLE <- 0

gen <- 60000
ind <- 0

all_data <- read.csv(file = paste0("N_1000_G_",gen,"_ind_",ind,"_initLE_",initLE,"_exp_quality_LS_",lifespan,"_envSD_",env_sd,"_envChange_",env_change,"_envChangeRate_",ch_rate,"_random.csv"), header = TRUE, sep = ',')

library(ggplot2)

# change working directly if needed
#setwd("")

# plotting re-scaled values

ggplot() + 
  geom_line(data = all_data, aes(x = cues, y = rescaled, colour = type), size = 2.0, alpha = 0.7)  +
  facet_wrap(~replicate) + 
  ylab("quality") + xlab("cue") +
  theme_bw() +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_line(colour="grey60", linetype="dashed"),
        panel.grid.minor.y = element_line(colour="grey60", linetype="dashed"),
        axis.title = element_text(size = 16), axis.text = element_text(size = 12),
               legend.position=c(1.87, 0.0),
        legend.text = element_text(size = 14)) +
  labs(colour='asd')

  ggsave(paste0("N_1000_G_",gen,"_ind_",ind,"_initLE_",initLE,"_exp_quality_rescaled_LS_",lifespan,"_envSD_",env_sd,"_envChange_",env_change,"_envChangeRate_",ch_rate,"_random_2.png")) 

