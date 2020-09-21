import keras
from keras.layers import Conv2D
from keras.models import Sequential
import numpy as np

def GMSD(x, y, c=0.0026):
    hx = np.array([[1/3,0,-1/3]]*3, dtype = np.float64)
    hy = hx.T
    hx = hx.reshape((1, 3, 3, 1, 1))
    hy = hy.reshape((1, 3, 3, 1, 1))

    modelx = Sequential()
    modelx.add(Conv2D(1,3,input_shape=(35,216,1),strides=1,bias=False,name='hx'))
    modely = Sequential()
    modely.add(Conv2D(1,3,input_shape=(35,216,1),strides=1,bias=False,name='hx'))
    modelx.set_weights(hx)
    modely.set_weights(hy)
    
    mr_sq = modelx.predict(x)**2 + modely.predict(x)**2
    md_sq = modelx.predict(y)**2 + modely.predict(y)**2
    mr = np.sqrt(mr_sq)
    md = np.sqrt(md_sq)
    GMS = (2*mr*md+c) / (mr_sq+md_sq+c)
    GMSM = np.mean(GMS)
    GMSD = np.sqrt(np.mean((GMS-GMSM)**2))
    return GMSD

if __name__ == "__main__":
    main()