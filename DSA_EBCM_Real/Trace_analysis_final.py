import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import seaborn as sns
import math


def Plot_chains(Traces, parameter_names=['Parameter'], colors=['orange'],size=[5,12],saving=False,file_name='Plot_chains.jpg', title='no_title'):
    dim= len(Traces)
    if len(parameter_names) != dim:
        parameter_names[0]=parameter_names[0] + '0'
        for i in range (1,dim):
            parameter_names.append('Parameter ' + str(i))

    if len(colors) != dim:
        colors=dim*['orange']
    
    fig, (ax) = plt.subplots(dim, 1,figsize=(size[0], size[1]))
    if title != 'no_title':
        fig.suptitle(title)

    for i in range (dim):
        ax[i].plot(Traces[i], color=colors[i])#marker='o',ms=2)
        ax[i].set_title(r'Inference of ' + parameter_names[i]) 
        ax[i].grid()

    
    plt.tight_layout()
    
    if saving:
        plt.savefig(file_name)

    plt.show()
    return

def Burn_in(Traces,burn_in=0):
    Distribs=[]
    for i in range (len(Traces)):
        Distribs.append(Traces[i][-burn_in:])
    return Distribs


def Postirior_median(Postiriors,burn_in=2000, R0=False):
    '''With postirors as obtained by the inference, returns  process a list of size len(parameters)
    which contains all the values of each parameter for all the inference process (as mirginal postiriors)
    '''
    Medians=[]
    param_dim=len(Postiriors[0])
    samples=len(Postiriors)
    Medians=[]
    for j in range (param_dim):
        meds=[]
        for i in range (samples):
            meds.append(np.nanmedian(Postiriors[i][j][-burn_in:]))
        
        Medians.append(meds)

    return Medians


def Plot_postirors(trace, parameter_names=None, colors=None,size=[5,12],saving=False,file_name='Plot_postirior.jpg',
                   title='no_title', real_values=None, with_mean=False,burn_in=0, given_space=False,with_density=False,with_alpha=None):
    
    
    if given_space:
        Traces=[]
        for i in range (len(trace[0])):
            Traces.append([x[i] for x in trace])
    else:
        Traces=trace
    
    
    dim= len(Traces)

    Distribs=Burn_in(Traces,burn_in)
    
    
    if parameter_names is None or len(parameter_names)!=dim:
        parameter_names=[]
        for i in range (0,dim):
            parameter_names.append('Parameter ' + str(i))


    if colors is None:
        colors=dim*['orange']
    
    fig, (ax) = plt.subplots(dim, 1,figsize=(size[0], size[1]))
    if title != 'no_title':
        fig.suptitle(title)

    for i in range (dim):
        ax[i].hist(Distribs[i], rwidth=0.9,bins=20, color=colors[i], label='Postirior',density=with_density, alpha=with_alpha)
        ax[i].set_title(r'Inference of ' + parameter_names[i]) 
        if with_mean:
            ax[i].axvline(x = np.mean(Distribs[i]), color = 'red', label = r'Expected value',lw=3)
        if real_values is not None:
            ax[i].axvline(x = np.mean(real_values[i]), color = 'black', label = r'Expected value',lw=3)
        ax[i].legend()


    
    plt.tight_layout()
    
    if saving:
        plt.savefig(file_name)

    plt.show()
    return

    
def Pair_plots(trace, parameter_names=None, color=None,size=[5,12],saving=False, file_name='Pair_plot.jpg', title='no_title', 
            burn_in=0, with_R0=False,statistics=False, given_space=False,with_alpha=None):

    if given_space:
        Traces=[]
        for i in range (len(trace[0])):
            Traces.append([x[i] for x in trace])
    else:
        Traces=trace
    
    dim= len(Traces)
    
    
    if parameter_names is None or len(parameter_names)!= dim:
        parameter_names=[]
        for i in range (0,dim):
            parameter_names.append('Parameter ' + str(i))

    if color is None:
        color='darkorange'
        
    if given_space:
        df = pd.DataFrame(trace[-(len(trace)-burn_in):],columns=parameter_names)
    else:
        df = pd.DataFrame(np.transpose(Traces),columns=parameter_names)

    if with_R0:
        R0=[]
        for i in range (len(Traces[0])):
            R0.append((Traces[2][i]-1)*Traces[0][i]/(Traces[0][i]+Traces[1][i]))

        df[r'$R_0$'] = R0
    
    from scipy.stats import pearsonr
    def reg_coef(x,y,label=None,color=None,**kwargs):
        ax = plt.gca()
        r,p = pearsonr(x,y)
        ax.annotate('r = {:.2f}'.format(r), xy=(0.5,0.5), xycoords='axes fraction', ha='center')
        ax.set_axis_off()
    
    
    g = sns.PairGrid(df,diag_sharey=False,height=1.6,aspect=1.2)
    g.map_diag(sns.kdeplot,color=color)
    g.map_lower(plt.scatter, color=color,s=1,alpha=with_alpha)
    g.map_upper(reg_coef,font_sizes=30)
    
    if saving:
        plt.savefig(file_name)

        
    plt.show()
    return

def Postirior_space(Postiriors,burn_in=0,check_conv=True):
    '''With postirors as obtained by the inference, returns  a list of  elements of size= inferenc_processes*chain_size
    each element is a point the parameter sapce, so each element is len(prameters) dimensional
    '''
    
    param_dim=len(Postiriors[0])
    samples=len(Postiriors)
    chain_lenght=len(Postiriors[0][0])
    Conv=True
    if burn_in<0:
        burn_in=chain_lenght+burn_in
        
    Postiriors_combined=[]
    for i in range (samples):
        if check_conv:
            for l in range (param_dim):
                if math.isnan(np.mean(Postiriors[i][l][burn_in:])) or math.isinf(np.mean(Postiriors[i][l][burn_in:])):
                    Conv=False
                    break
            if Conv==False: 
                continue
                
        for j in range (burn_in,chain_lenght):
            dot=[]
            for k in range (param_dim):
                dot.append(Postiriors[i][k][j])
            Postiriors_combined.append(dot)
            
    return np.array(Postiriors_combined)


def convert_p_k(data, s0=False, space=True):
    if space==True:
        New_space=[]
        for point in data:
            if s0:
                k=point[2]*(1-point[-2])/point[-2]
            else:
                k=point[3]+point[2]*(1-point[-2])/point[-2]
            New_space.append([point[0],point[1], k, point[2], point[3], point[-1]])
        return np.array(New_space)
    else:
        New_Marginal=[]
        New_Marginal.append(data[0])
        New_Marginal.append(data[1])
        #calculation of k
        if s0:
            ks=k_from_rps(data[2],data[-2], data[3])
        else:
            ks=k_from_rps(data[2],data[-2], len(data[2])*[0])
        New_Marginal.append(ks)
        #r and shift
        New_Marginal.append(data[2])
        New_Marginal.append(data[-2])
        #rho
        New_Marginal.append(data[-1])
        return New_Marginal
        
def k_from_rps(r,p,s):
    k=[]
    for i in range (len(r)):
        k.append(s[i]+(r[i]*(1-p[i])/p[i]))
    return k

def Box_plots(Postiriors_cases,cases=['no_labels'], parameter_names=None, colors=None, sizes=[4,15],saving=False,file_name='Plot_box_no_name.jpg',
                   title=None, real_values=None, with_patch_artist=False, with_alpha=None,with_notch=False,with_showfliers=False,with_mean=False):
    
    para_dim=len(Postiriors_cases[0])
    
    if parameter_names is None:
        parameter_names=[]
        for i in range (0,para_dim):
            parameter_names.append('Parameter ' + str(i))


    print(len(Postiriors_cases),len(Postiriors_cases[0]),len(Postiriors_cases[0][0]))
    
    fig, (ax) = plt.subplots(para_dim, 1,figsize=(sizes[0], sizes[1]))
    
    for j in range (para_dim):
        box=[]
        for i in range (len(Postiriors_cases)):
            box.append(Postiriors_cases[i][j])
        print('probelm',len(box),len(cases))
        bp=ax[j].boxplot(box, labels = cases,patch_artist=with_patch_artist,notch=with_notch,showfliers=with_showfliers)

        if with_patch_artist:
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                if with_alpha is not None:
                    patch.set_alpha(with_alpha)
            for median in bp['medians']:
                median.set_color('black')
            
        if with_mean:
            ax[j].axhline(np.mean(Postiriors_cases[i][j]), color='green', label='Mean')
        ax[j].set_title(r'Inference of ' + parameter_names[j])
        if real_values is not None:
            ax[j].axhline(real_values[j], label='real value')
            
    plt.tight_layout()
    plt.legend()
    if saving:
        plt.savefig(file_name)
    
    return



def bar_3d(x,y,Title='Title', y_label='y', x_label='x', c_map='viridis',saving=False,bins=50):
    from matplotlib import cm

    fig = plt.figure()          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')
    
    #make histogram stuff - set bins - I choose 20x20 because I have a lot of data
    hist, xedges, yedges = np.histogram2d(x, y, bins=(50,50))
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
    
    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like (xpos)
    
    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()
    
    cmap = cm.get_cmap(c_map) # Get desired colormap - you can change this!
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    plt.title(Title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

    if saving:
        plt.savefig(file_name)


    return 


def get_single_estimation(PS, method='mean', Tmax=10, extend=False,band=0,steps_ode=3000, func='EBCM'):
    estimatiors=[]
    dim = len(PS[0])
    if method=='mean':
        for i in range (dim-1):
            estimatiors.append(np.nanmean(PS[:,i]))
        estimatiors.append(np.nanmean(PS[:,-1]))
    elif method=='median':
        for i in range (dim-1):
            estimatiors.append(np.nanmedian(PS[:,i]))
        estimatiors.append(np.nanmedian(PS[:,-1]))
    elif method=='mode':
        for i in range (dim-1):
            estimatiors.append(get_point_max_probability(PS[:,i]))
        estimatiors.append(np.nanmedian(PS[:,-1]))
    elif method=='density_ms':
        estimatiors=density_Mean_shift(PS[:,0:dim],band=band)
        
    if extend:
        estimatiors=extend_p_var_R0(estimatiors)
    print(method, estimatiors)
    if func=='CM':
        S,I,R = ode_CM(estimatiors,Tmax=T,steps_ode=steps_ode)
        times_mf=np.linspace(0,T,steps_ode+1)  
        return [S,I,R,times_mf], estimatiors
    elif func=='EBCM':
        thetas,S,I,R,times,Cum_I,new_I=ode_EBCM(estimatiors,dS_EBCM,psi_NB,d1_psi_NB,Tmax=Tmax,steps_ode=10000,Complete=True)
        return [S,I,R,times], estimatiors

def extend_p_var_R0(estimators):
    estim=[]
    ps=estimators[3]/(estimators[3]+estimators[2])
    var_s=estimators[3]*(1-ps)/(ps**2)
    R0_s=(estimators[0]/(estimators[0]+estimators[1]))*((var_s+estimators[2]**2-estimators[2])/(estimators[2]))
    estim.append(estimators[0])
    estim.append(estimators[1])
    estim.append(estimators[2])
    estim.append(estimators[3])
    estim.append(ps)
    estim.append(var_s)
    estim.append(R0_s)
    estim.append(estimators[-1])
    return estim


def get_point_max_probability(sample):
    hist, bins = np.histogram(sample, bins='auto', density=True)
    # Calculate the bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    max_=np.max(hist)
    index=np.where(hist == max_)
    return bin_centers[index]

def Min_square_error(x,y):
    mse=0
    if len(x) != len(y):
        return 'Problem size'
    else:
        for i in range (len(x)):
            mse+=(x[i]-y[i])**2
    return np.sqrt(mse)/len(x)

def get_sample(times,I,size=10):
    sample=range(0,len(times),500)
    ntimes=[]
    nI=[]
    for s in sample:
        ntimes.append(times[s])
        nI.append(I[s])
    return ntimes,nI

def density_Mean_shift(PS,thining=False,jump=50, band=0):
    from sklearn.cluster import MeanShift
    from collections import Counter
    if thining:
        thinned_PS=thining_chain(PS,jump=jump)
    else:
        thinned_PS=PS
    print(len(thinned_PS))
    if band==0:
        clustering = MeanShift().fit(thinned_PS)
    else:
        clustering = MeanShift(bandwidth=band).fit(thinned_PS)
    labels=clustering.labels_
    dic_labels=Counter(labels)
    index_final=list(dic_labels.keys())[list(dic_labels.values()).index(max(dic_labels.values()))]
    print('Max cluster',max(dic_labels.values()))
    print('Number clusters', len(dic_labels.keys()))
    estim=list(clustering.cluster_centers_[index_final])
    return estim

def cluster_sizes(centroids, data):
    
    centroid_belong={}
        
    for j in range (len(data)):
        d=data[j]
        dist_to_centroids={}
        for i in range (len(centroids)):
            dist_to_centroids[i]=np.linalg.norm(d-centroids[i])

        centroid_belong[j]=list(dist_to_centroids.values()).index(min(dist_to_centroids.values()))
    cluster_size={}
        
    for j in centroid_belong.keys():
        cluster=centroid_belong[j]
        if cluster not in cluster_size.keys():
            cluster_size[cluster]=1
        else:
            cluster_size[cluster]+=1
    return cluster_size

def get_Incidence(para,K,Tmax):
    t0=0; theta0=1e-6
    thetas,S,I,R,times,Cum_I,new_I=ode_EBCM(para,dS_EBCM,psi_NB,d1_psi_NB,Tmax=Tmax,steps_ode=20000,Complete=True)
    N_eff=K/(1-S[-1])
    #print(N_eff)
    days=range(Tmax)
    S_days=Days_value(days,times,S)
    new_Infec=[]
    for i in range (1,len(S_days)):
        new_Infec.append(S_days[i-1]-S_days[i])
    
    new_Infec=np.array(new_Infec)
    #plt.plot(times,S)
    
    return new_Infec,N_eff


def Get_mean_CI_from_sample(estimators,days,sample=1000,Tmax=100):
    import scipy.stats as st
    import math
    import random 
    K_real=len(infection_times)
    Tmax=Tmax
    var_K=420
    sample_size=0
    theta0=1e-8
    estimators_clean=[]
    Est_I=[];Est_R=[];Est_nI=[];Est_cI=[];Est_S=[];Est_N=[]

    if specefic_sample:
        while sample_size<sample:
            estim=random.choice(estimators)
            #estim[-1]=np.median(estimators[:,-1])
            thetas,S,I,R,times_mf,Cum_I,new_I=ode_EBCM(estim,dS_EBCM,psi_NB,d1_psi_NB,Tmax=Tmax,steps_ode=10000,theta0=theta0,Complete=True)
            K=random.choice(range(K_real-var_K,K_real+var_K))
            #S_T=random.choice(np.linspace(-0.05,0.05,50))
            N=K/(1-(S[-1]))
            if N*estim[-1]>5 or math.isnan(np.mean(I)) :continue
            Est_I.append(Days_value(days,times_mf,I*N))
            Est_R.append(Days_value(days,times_mf,R*N))
            Est_cI.append(Days_value(days,times_mf,Cum_I*N))
            Est_S.append(Days_value(days,times_mf,S*N))
            Est_N.append(N)
            #estim=[estim[0],estim[1],estim[2],estim[3],estim[4],estim[5],N,estim[-2],estim[-1]]
            estimators_clean.append(estim)
            print(sample_size,K,N)
            sample_size+=1       
    newI=[]
    for S in Est_S:
        newI=[0]
        for i in range (1,len(S)):
            newI.append(S[i-1]-S[i])
        Est_nI.append(newI)
    
    Estim_dyn=[np.array(Est_I),np.array(Est_nI),np.array(Est_cI),np.array(Est_R)]

    CI_top=[]
    CI_down=[]
    mean_dyn=[]
    for estim in Estim_dyn:
        top=[]
        down=[]
        mean=[]
        for d in days:
            Order_list=sorted(estim[:,d])
            mean.append(np.nanmean(estim[:,d]))
            top.append(Order_list[round(2.5*sample_size/100)])
            down.append(Order_list[int(97.5*sample_size/100)])
        print(len(top))
    
        CI_top.append(np.array(top));CI_down.append(np.array(down));mean_dyn.append(np.array(mean));
        
    return CI_top,CI_down,mean_dyn, Est_N, np.array(estimators_clean)

def Days_value(dyas, times, list_):
    new_list = np.column_stack((times, list_))
    days_list= interp1d(days, new_list)
    return days_list[:,1]

############################# For solving the ODEs ###########################################

def psi_Poi(theta,para):
    mu=para[2]
    res=np.exp(mu*(theta-1))
    return res

def d1_psi_Poi(theta,para):
    mu=para[2]
    res=mu*np.exp(mu*(theta-1))
    return res


def psi_Reg(theta,para):#,shift=3):
    a=para[2]
    res=theta**(a)
    return res

def d1_psi_Reg(theta,para):#,shift=3):
    a=para[2]
    res=a*theta**(a-1)
    return res 


def psi_NB(theta,para):#,shift=3):
    r=para[2]
    shift=para[3]
    p=para[-2]
    
    #p=r/(r+k-shift)
    res=(theta**(shift))*(p/(1-(1-p)*theta))**r
    return res

def d1_psi_NB(theta,para):#,shift=3):
    r=para[2]
    shift=para[3]
    p=para[-2]
    
    A=shift * (theta**(shift-1)) * (p/(1-(1-p)*theta))**r 
    B=(theta**shift)*(r*(1-p)*p**r)/((1-(1-p)*theta)**(1+r))
    res = A + B
    return res 

def psi_NB_k(theta,para):#,shift=3):
    r=para[3]
    k=para[2]
    shift=para[4]
    
    p=r/(r+k-shift)
    
    res=(theta**(shift))*(p/(1-(1-p)*theta))**r
    return res

def d1_psi_NB_k(theta,para):#,shift=3):
    r=para[3]
    k=para[2]
    shift=para[4]
    
    p=r/(r+k-shift)
    
    A=shift * (theta**(shift-1)) * (p/(1-(1-p)*theta))**r 
    B=(theta**shift)*r*(p/(1-(1-p)*theta))**(r-1) 
    C=(1-p)*p/((1-(1-p)*theta)**2)
    res = A + B*C
    return res     

def ode_S_EBCM(para,dS,psi,d1_psi,Tmax):
    dt=Tmax/5000
    b = para[0]; g = para[1]; ic = 1-1e-6; rho = para[-1]

    n = int(Tmax/dt) + 1
    xmat = np.zeros(n)
    x = ic
    xmat[0] = x
    for i in range(1, n):
        x = xmat[i-1]
        sol = -b * x + b *(1-rho)*d1_psi(x, para) / d1_psi(1, para) + g *(1-x)
        xmat[i] = xmat[i-1] + sol * dt
    return np.array(xmat)

def ode_R_EBCM(para,S, Tmax):
    dt=Tmax/5000
    b = para[0]; g = para[1]; ic = 1-1e-6; rho = para[-1]

    n = int(Tmax/dt) + 1
    xmat = np.zeros(n)
    x = 0
    xmat[0] = x
    for i in range(1, n):
        x = xmat[i-1]
        sol = g*(1-S[i-1]-x)
        xmat[i] = xmat[i-1] + sol * dt
    return np.array(xmat)

'''dS''' 
def dS_EBCM(theta, para, psi,d1_psi):
    b=para[0]; g=para[1]; rho = para[-1]
    res = d1_psi(theta, para)*(-b * theta + b* (1-rho)*d1_psi(theta, para) / d1_psi(1,para) + g *(1-theta))
    if np.max(res)>0:
        res = res - np.max(res) - 1e-100
    return res

def Solve_EBCM(psi,d1_psi,parameters, Tmax=16):
    
    times=np.linspace(0,Tmax,3001)

    para=parameters
    gamma=para[1]
    thetas=ode_S_EBCM(para,dS_EBCM,psi,d1_psi,Tmax=Tmax)
    S_EBCM=psi(thetas,para)
    R_EBCM=ode_R_EBCM(para,S_EBCM, Tmax=Tmax)
    I_EBCM=gamma*(1-S_EBCM-R_EBCM)
    
    return S_EBCM,I_EBCM, R_EBCM
