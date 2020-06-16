import tensorflow as tf
import tensorflow.keras as KK

class Model():

    def __init__(self, args):

        self.args = args

        #### inputs
        inputs = KK.layers.Input(shape=(args.rows,args.cols,args.baseinfo), name="inputs") # [None, 16, 640, 4] The MSA
        """
        - binfo=integer base("acgt-")+msa: missing=0, refAreadA=1, refAreadC=2,
        refAreadG=3, refAreadT=4, refAread-=5, refCreadA=6, ..., ref-read-=25,
        overflow=26. This will indicate match, insert(ref-)...
        
        - snr, ipd, pw ( 3 numbers )
        """

        #### take the binfo and embed 3+4=7
        baseint = KK.layers.Lambda( lambda xx: xx[:,:,:,0] )(inputs)
        embed = KK.layers.Embedding(input_dim=26+1, output_dim=4, mask_zero=False, name="embed")(baseint)
        merged = KK.layers.Concatenate(axis=3,name="merged")([inputs,embed]) # (?, 16, 640, 8)
        perm = KK.layers.Lambda( lambda xx: KK.backend.permute_dimensions( xx, (0,2,1,3)), name="perm")(merged) # (?, 640, 16, 8)
        reshape = KK.layers.Lambda( lambda xx: KK.layers.Reshape((args.cols, args.rows*8))(xx), name="rehape")(perm) # (?, 640, 128)

        #### RNN (?, 640, rnn_hidden_size)
        rnn_hidden_size= 64

        #### The RNN layers
        # 0th RNN layer
        RNNNotBack0 = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=False, name="RNNNotBack0")(reshape)
        RNNYesBack0 = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=True, name="RNNYesBack0")(reshape) 

        # 1st RNN layer
        #RNNNotBack1 = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=False, name="RNNNotBack1")(RNNNotBack0)
        #RNNYesBack1 = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=True, name="RNNYesBack1")(RNNYesBack0)

        # last RNN layer
        RNNNotBack = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=False, name="RNNNotBack")(RNNNotBack0)
        RNNYesBack = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=True, name="RNNYesBack")(RNNYesBack0)

        rnnconcat = KK.layers.Concatenate(axis= -1,name="rnnconcat")([RNNNotBack, RNNYesBack]) # [None, 640, 128]

        #### make predications on direct base
        pred0=  KK.layers.Dense(64, activation='elu',name="pred0")(rnnconcat)
        pred =  KK.layers.Dense(5, activation='softmax',name="pred")(pred0) # [None, 640, 5]

        ################################
        self.model = KK.models.Model(inputs=[inputs], outputs=[pred]) 
        
        #self.model.summary()
        print("================================")
        for layer in self.model.layers:
            print("layer: name",layer.name, "outputshape", layer.get_output_at(0).get_shape().as_list())
        print("================================")

        #myopt = KK.optimizers.SGD()
        myopt = KK.optimizers.Adam()

        # "kullback_leibler_divergence" "sparse_categorical_crossentropy"
        # loss_weights=[0.0,1.0,0.0,0.0])
        self.model.compile(optimizer=myopt, 
                           loss=["sparse_categorical_crossentropy"])
                           
