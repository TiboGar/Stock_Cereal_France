# base
import zipfile
import os
import requests

# dataframe
import pandas as pd 
import geopandas as gpd
import numpy as np
import warnings

class famDataset:
    def __init__(self, root_dir: str):
        """
        Initializes the DatasetManager with the given root directory.
        
        Parameters:
        - root_dir (str): The path to the directory where the dataset is stored.
        """
        self.root_dir = root_dir
        self.geom_dir = 'FrenchDep'
        self.raw_data = None

        self.transformed_data = None
        self.transformed_pivot_to_long = None
        self.transformed_convert_attributes = None

        self.geom_data = None
        self.st_matrix = None

    def download_raw_data(self, url='https://visionet.franceagrimer.fr/Pages/OpenDocument.aspx?fileurl=SeriesChronologiques%2fproductions%20vegetales%2fgrandes%20cultures%2fcollecte%2cstocks%2cd%c3%a9p%c3%b4ts%2fSCR-GRC-histDEP_collecte_stock_depuis_2000-C25.zip&telechargersanscomptage=oui'):
        """
        Downloads a zip file from the specified URL, extracts it,
        and saves the contents to the root directory.

        Parameters:
        - url (str): The URL from which to download the zip file. Defaults to 'https://visionet.com'.

        Returns:
        - extracted_path (str): The path to the directory of the extracted files.
        """

        # Define the zip file name
        zip_file_name = os.path.join(self.root_dir, 'dataset.zip')

        # Download the zip file
        response = requests.get(url)
        
        if response.status_code == 200:
            # Write the zip file
            with open(zip_file_name, 'wb') as zip_file:
                zip_file.write(response.content)

            # Extract the zip file
            extracted_dir = os.path.join(self.root_dir, 'extracted_data')
            with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)

            print(f"Download and extraction complete: {extracted_dir}")
            return extracted_dir
        else:
            print(f"Failed to download from {url}: Status code {response.status_code}")
            return None
        
    def load_raw_data(self):
        """
        Load raw data from the extracted_data directory
        
        Returns:
        - None
        
        Set self.attributes:
        - raw_data
        """
        
        folderPath = os.path.join(self.root_dir, 'extracted_data')
        filePath =  [file for file in os.listdir(folderPath) if "C25.csv" in file][0]
        filePath = os.path.join(folderPath,filePath)

        # Format is special: European decimal with ISO-8859-1 encoding
        self.raw_data = pd.read_csv(filePath, sep=";",encoding = "ISO-8859-1", decimal = ",",low_memory=False )


        print("Raw data loading complete")
        return None
    
    def load_geometry_data(self, geom_dir = 'FrenchDep'):
        """
        Load geometry data from dir directory
        French departments are open source data with departement shape and multiple id (name, number, ...)

        Returns:
        - None

        Set self.attributes:
        - geom_data
        """        
        self.geom_dir = geom_dir
        file_path = os.path.join(os.path.join(self.root_dir,self.geom_dir),'departements-20140306-100m-shp')
        france = gpd.read_file(file_path, encoding = 'iso-8859-15')

        # Suppress department outside France metropolitaine 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # Compute the centroids 
            france["centroid"] = france.centroid
            # compute their distance from the middle of France 
            depDistance = france['centroid'].distance(france.loc[france['nom'] == 'Allier','centroid'].iloc[0])
            # remove departements when distance is large (dom-tom departments) 
            france = france[(depDistance < np.std(depDistance)*2)]

        # drop not used columns:
        france.drop(['nom','nuts3','wikipedia'], axis = 1, inplace=True)

        # reduce geometry complexity to reduce plot time
        france.geometry = france.geometry.simplify(tolerance= 0.05)

        self.geom_data = france

        print("Geom data loading complete")
        return None

    def load_transformed_data(self, convert_attributes = True, drop_attributes = True, pivot_to_long = False):
        """
        load raw data and transform it:
        - cleaning
        - attributes transformation (y/n)
            The five attributes are converted to ENTREE and SORTIE, eluding the complexity in simple metrics
        - pivot to long (y/n)
            The data is set in a tidy long format
        
        Returns:
        - None
        
        Set self.attributes:
        - transformed_data
        - transformed_pivot_to_long
        - transformed_convert_attributes
        """
        ### loading ### 
        # raw_data
        if(self.raw_data is None): 
            self.load_raw_data()

        # --------------------------------------- #
        ### Correct errors and processing ###
        # --------------------------------------- #
        
        df = self.raw_data.copy()
        
        # FranceAgrimer gère ses bases de données n'importe comment.
        # Ils sont tellement terribles, je suspecte que ce soit volontaire. 
        ## Error 0 - Correct the renaming frenzy 
        df.rename(columns={'NUMERO_DEPARTEMENT': 'DEP','NOM_DEPARTEMENT':'DEPARTEMENT','NOM_REGION':'REGION'}, inplace=True)

        ## Error 1 - Remove whitespace
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()
        
        ## Error 2 - The database contain >240 errors on the ANNEE field. 
        # The field CAMPAGNE is used instead when ANNEE is incorrect.
        # Correct ANNEE
        df['CAMPAGNE_NUM'] = df['CAMPAGNE'].str[:4]
        df['CAMPAGNE_NUM'] = pd.to_numeric(df['CAMPAGNE_NUM'], errors='coerce')

        # if the month is between 1 and 6 the CAMPAGNE is delayed by 1 year to ANNEE
        df['TRUE_ANNEE'] = df['CAMPAGNE_NUM'] + (df['MOIS'] <7)*1

        TriggerErrorANNEE = (df['TRUE_ANNEE'] == df['ANNEE']) == False
        df.loc[TriggerErrorANNEE,'ANNEE'] = df['TRUE_ANNEE']

        ## create DATE from ANNEE + MOIS
        df['DATE'] = pd.to_datetime(pd.DataFrame({'year' : df['ANNEE'],
                                                  'month' : df['MOIS'],
                                                  'day' : 1}))
        
        ## Error 4 - Duplicates are observed
        # I believe duplicates come from manual sum of rows. Indeed FAM sum every actor departement wise.
        # Therefore my proposed treatment of duplicates is the summation of them. 
        
        value_columns = df.select_dtypes('float').columns

        # Sum duplicates together
        df[value_columns] = df.groupby(['ESPECES','DEP','CAMPAGNE','DATE'])[value_columns].transform('sum')
        # and then drop them
        df = df.drop_duplicates(subset= ['ESPECES','DEP','CAMPAGNE','DATE'])
        
        UT_dup = df.duplicated().sum() == 0

        # Corrected by Error 4 
        ## Error 3 - Spurious null STOCKS are observed
        # A detection and inference function is set, see below the class
        # detection boolean: STOCKS == 0 and prev_ > 100 and next_ > 100
        # Inference is realised with mean of previous and next stock. 

        #df[['DETECT','INFER']] = detect_and_infer(df)
        
        # Replace outlier stocks 
        #df.loc[df['DETECT'],'STOCKS'] = df.loc[df['DETECT'],'INFER'] 

        ## drop not used columns 
        df.drop(['REGION','Unnamed: 13','DEPARTEMENT','ANNEE','MOIS','TRUE_ANNEE','CAMPAGNE_NUM'],axis=1, inplace=True)

        # --------------------------------------- #
        ### attributes conversion ###
        # --------------------------------------- #
        
        if(convert_attributes):
            # farmers_entry = TOTAL_COLLECTE + ENTREE_DEPOT
            df['farmers_entry'] = df['TOTAL_COLLECTE']+ df['ENTREE_DEPOT']

            # compute Var(STOCKS) the first lag difference of the STOCKS column for each combination of ESPECES and DEP
            df['LAG_DIFF'] = df.groupby(['ESPECES', 'DEP'])['STOCKS'].transform(lambda x: x - x.shift(1)).round(3).fillna(0)

            # movement = Var(STOCKS) - TOTAL_COLLECTE - SORTIE_DEPOT - REPRISE_DEPOT
            df['movement'] = df['LAG_DIFF'] - df['TOTAL_COLLECTE'] - df['SORTIE_DEPOT'] - df['REPRISE_DEPOT']

            # Due to precision issues, there are some number 10 zeros after the . I delete them
            df[['farmers_entry','movement','LAG_DIFF']] = df[['farmers_entry','movement','LAG_DIFF']].round(3)
            
            df['stocks'] = df['STOCKS'] + df['STOCKS_DEPOTS']
            
            # drop previous attributes
            if(drop_attributes):
                df.drop(['CAMPAGNE', 'TOTAL_COLLECTE', 'STOCKS', 'STOCKS_DEPOTS', 'ENTREE_DEPOT',
                        'SORTIE_DEPOT', 'REPRISE_DEPOT', 'LAG_DIFF'], axis = 1, inplace=True)
            

        ### pivot dataframe to long ### 
        if(pivot_to_long):
            value_vars = df.select_dtypes('float')
            df = pd.melt(df, id_vars=['ESPECES','DEP','DATE','CAMPAGNE'], value_vars= value_vars)


        self.transformed_data = df
        self.transformed_convert_attributes = convert_attributes
        self.transformed_pivot_to_long = pivot_to_long

        print("Raw data processed in transformed_data")
        return None
    

    def save_transformed_data(self, path = ''):
        self.load_transformed_data(convert_attributes = True, drop_attributes = True, pivot_to_long = True)   
        save_path = os.path.join(self.root_dir,path,'fam_transform.csv')
        self.transformed_data.to_csv(save_path)

        print("transformed_data saved in a long format in csv")
        return None


def make_spatiotemporal_matrix(transformed_data, geom_data, especes, variable ):
    """
    merge raw data and geom data in a matrix (DEP, DATE)
    - with DEP the number of unique department
    - DATE the number of unique DATE 
    ! DATE will vary as dataset expand

    ESPECES and ATTRIBUTES must be selected by user

    Returns:
    - st
    """    
    ### select ESPECES and ATTRIBUTES
    if(not especes in transformed_data['ESPECES'].unique()):        
        print('Available values :',transformed_data['ESPECES'].unique())
        raise ValueError("Error: especes is not found.")
        
    if(not variable in transformed_data.select_dtypes('float').columns):        
        print('Available values :',transformed_data.select_dtypes('float').columns)
        raise ValueError("Error: variable is not found.")
        
    st = transformed_data.loc[transformed_data['ESPECES'] == especes,['DEP', 'DATE', variable]]
    
    ### pivot to wide
    st = pd.pivot_table(st, index=['DEP'], columns='DATE', values=variable).fillna(0)

    ### merge with geometry and convert to geopandas DataFrame ###
    geom_data['DEP'] = np.array(geom_data['code_insee'])

    st = geom_data.merge(st, on='DEP')

    # delete useless columns
    st.drop(['code_insee'], axis = 1, inplace=True)

    return st

def detect_and_infer(df):
    # copy
    grp = df.groupby(['ESPECES', 'DEP'])['STOCKS']
    prev_ = grp.shift(1)
    next_ = grp.shift(-1)
    # detection boolean: STOCKS == 0 and prev_ > 100 and next_ > 100
    detect = (df['STOCKS'] == 0) & (prev_ > 100) & (next_ > 100)
    # inference: mean of prev_ and next_ when detected, else NaN
    infer = np.where(detect, (prev_ + next_) / 2.0, np.nan).round(0)
    # return DataFrame with only the two requested columns (lowercase names)
    return pd.DataFrame({
        'detect': detect.astype(bool).values,
        'infer': infer
    }, index=df.index)
