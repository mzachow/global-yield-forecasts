#!/usr/bin/env python3

"""
This module contains the definition of the YieldDataset class for reading and preparing crop yield datasets.
"""
import pandas as pd

class Dataset:
    """
    A custom dataset class to handle crop yield dataset from various sources.
    """
    
    def __init__(self, path, country, crop, quantity, accurarcy="irrelevant"):
        data = self.load_data(path, country, crop, quantity)
        processed_data = self.preprocess_data(data, country, quantity, accurarcy)
        self.dataset = processed_data

    def load_data(self, path, country, crop, quantity):
        """
        reads and filters dataset based on user input. 
        """
        df = pd.read_csv(path, encoding="utf-8-sig")
        countries = df["Area"].unique()
        
        # try to filter by country
        try:
            df = df.loc[df["Area"].str.lower() == country.lower()].reset_index(drop=True)
            if df.empty:
                raise RuntimeError("Country doesn\'t exist")
        except Exception as e:
            print("The country doesn\'t exist, must be one of".format(e) + str(countries))
        
        # crops that are cultivated in that country 
        crops = df["Item"].unique()    
        # try to filter by crop
        try:
            df = df.loc[df["Item"].str.lower() == crop.lower()].reset_index(drop=True)
            if df.empty:
                raise RuntimeError("Crop doesn\'t exist")
        except Exception as e:
            print("The crop doesn\'t exist for the selected country, must be one of".format(e) + str(crops))    
        
        # quantities for the selected crop-country pair
        quantities = df["Element"].unique()  
        # try to filter by quantity
        try:
            df = df.loc[df["Element"].str.lower() == quantity.lower()].reset_index(drop=True)
            if df.empty:
                raise RuntimeError("Quantity doesn\'t exist")
        except Exception as e:
            print("The quantity doesn\'t exist for the selected crop-country pair, must be one of".format(e) + str(quantities)) 
        
        return df
    
    def preprocess_data(self, data, country, quantity, accuracy):
        """
        perform basic preprocessing to the data
        """
        
        # drop irrelevant columns
        data = data[[c for c in data.columns if c not in [
            'Area Code', 'Area Code (M49)', 'Item Code', 'Item Code (CPC)', 'Element Code']]]
        
        # store unit
        unit = data["Unit"].unique()[0]
        
        # flags and values are stored in columns for each year
        df_flag = data[[c for c in data.columns if ("Y" in c) & ("F" in c)]].transpose().reset_index()
        df_flag["index"] = df_flag["index"].str[+1:-1].astype(int)
        df_flag.columns = ["year", "flag"]
        df_value = data[[c for c in data.columns if ("Y" in c) & ("F" not in c)]].transpose().reset_index()
        df_value["index"] = df_value["index"].str[+1:].astype(float)
        df_value.columns = ["year", "value"]
        
        data = df_value.merge(df_flag, on="year")
        
        if accuracy != "irrelevant":
            # accuracy flags for the selected quantity of the crop-country pair
            flags = data["flag"].unique()  
            # try to filter by quantity
            try:
                data = data.loc[data["flag"] == accuracy.upper()[0]].reset_index(drop=True)
                if data.empty:
                    raise RuntimeError("Accuracy level doesn\'t exist")
            except Exception as e:
                print("The accuracy level doesn\'t exist for the selected quantity of the crop-country pair, must be one of".format(e) + str(flags)) 
         
        data = data.assign(unit = unit)   
        if unit == "hg/ha":
            data[["value"]] /= 1e4   
            data = data.assign(unit = "t/ha")
        if unit == "ha":
            data[["value"]] /= 1e6   
            data = data.assign(unit = "mio. ha")
        if unit == "tonnes":
            data[["value"]] /= 1e6   
            data = data.assign(unit = "mio. t")
                    
        data["value"] = data["value"].round(2)
        data = data.assign(quantity = quantity)
        data = data.assign(country = country)
        
        return data
    
    def _filter_by_years(self, year_range):
        df = self.dataset.copy()
        df = df.loc[df["year"].isin(year_range)].reset_index(drop=True)
        
        return df