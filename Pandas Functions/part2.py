import pandas as pd


def convert_People_Cell(cell):
    if cell == "n.a.":
        return 'sam Walton'
    return cell

def convert_Eps_Cell(cell):
    if cell == "not Available" : 
        return None 
    return cell


df = pd.read_csv('stock_data.csv',converters={
    'people': convert_People_Cell,
    'eps':  convert_Eps_Cell
})

# todo Write it to Excel File
df.to_excel('new.xlsx',sheet_name='stocks',startrow = 0,startcol=2)

# todo:;You Can Handle Other Separator By the Method (Sep): 