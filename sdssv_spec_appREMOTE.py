import dash 
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from astropy.io import fits
import requests
import io
import json
import numpy as np
import plotly.express as px
import plotly.colors
import pandas as pd

###
### input the data directory path 
###

#NOTE TO CODER: JSON LIKES STRING KEYS FOR DICTIONARIES!!!!!!
programs, plateIDs, catalogIDs = json.load(open("dictionaries.txt"))
authen = './authentication.txt'

### css files
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', \
'//maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css', \
'http://aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.css', \
'https://use.fontawesome.com/releases/v5.13.0/css/all.css', \
'https://use.fontawesome.com/releases/v5.13.0/css/v4-shims.css', \
]

###
### Any necessary functions
###

def SDSSV_buildURL(plateID, MJD, objID):
    """
    A function to build the url that will be used to fetch the data. 
    
    Catalog IDs don't start with zero but the URL needs it too,
    using zfill(11) fixes this.
    """
    url = "https://data.sdss5.org/sas/sdsswork/bhm/boss/spectro/redux/v6_0_2/" \
    +"spectra/lite/{}p/{}/spec-{}-{}-{}.fits".format(str(plateID), str(MJD), str(plateID), str(MJD), str(objID).zfill(11))
    
    return url

def SDSSV_fetch(username, password, plateID, MJD, objID):
    """
    Fetches spectral data for a SDSS-RM object on a 
       specific plate on a specific MJD. Uses the user
       supplied authentication. 
    
    TO DO: allow for multiple MJDs and plates, for loop it up
    """
    url = SDSSV_buildURL(plateID, MJD, objID)
    r = requests.get(url, auth=(username, password))  
    data_test = fits.open(io.BytesIO(r.content))
    flux = data_test[1].data['FLUX']
    wave = 10**data_test[1].data['loglam']
    return wave, flux 
'''
def fetch_catID(catID, plate):
    fluxes = []
    waves = []
    names = []
    for i in catalogIDs[str(catID)]:
        if plate == "all":
            dat = SDSSV_fetch(username, password,i[0], i[1], catID)
            fluxes.append(dat[1])
            waves.append(dat[0])
            names.append(i[3])
        else:
            if i[0] == plate:
                dat = SDSSV_fetch(username, password,i[0], i[1], catID)
                fluxes.append(dat[1])
                waves.append(dat[0])
                names.append(i[3])
            else:
                continue
    return waves, fluxes, names
'''
def fetch_catID(catID, plate):
    fluxes = []
    waves = []
    names = []
    for i in catalogIDs[str(catID)]:
        if plate == "all":
            dat = SDSSV_fetch(username, password,i[0], i[1], catID)
            fluxes.append(dat[1])
            waves.append(dat[0])
            names.append(i[3])
        else:
            if i[0] == plate:
                dat = SDSSV_fetch(username, password,i[0], i[1], catID)
                fluxes.append(dat[1])
                waves.append(dat[0])
                names.append(i[3])
            else:
                continue
    df = pd.DataFrame(fluxes,index=names,columns=waves[0])
    df = df.sort_index()
    return df

### set up color scale
def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:
    
        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    if intermed <= 0 or len(colorscale) == 1:
        return colorscale[0][1]
    if intermed >= 1:
        return colorscale[-1][1]

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    # noinspection PyUnboundLocalVariable
    return plotly.colors.find_intermediate_color(
        lowcolor=low_color, highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb")

viridis_colors, _ = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Inferno)
colorscale = plotly.colors.make_colorscale(viridis_colors)
    
###
### Authentication
###    
try:
    print("Reading authentication file.")
    with open(authen,'r') as i:
        lines = i.readlines()
        username = lines[0][:-1] #there will be a \n on the username
        password = lines[1][:-1]
except: #any error from above will fall through to here.
    print("authentication.txt not provided or incomplete. Please enter authentication.")
    username = input("Enter SDSS-V username:")
    password = input("Enter SDSS-V password:") 

try:
    print("Verifying authentication...")
    fetch_test = SDSSV_fetch(username, password, 15173, 59281, 4350951054)
    print("Verification successful.")
except:
    print("Authentication error, please cntrl-c and fix authentication.txt.")
    print("Contact Meg (megan.c.davis@uconn.edu) is the issue persists.")
 

    
### important spectra lines to label in plots
spectral_lines = { 'Ha': [6564], 
                   'Hb': [4862],
                   'MgII': [2798],
                   'CIII': [1908],
                   'CIV': [1549],
                   'Lya': [1215],}

### wavelength plotting range
wave_min = 3750
wave_max = 11000

### starting the dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

### get object info
### organize by program, plateid, catalogid
programname = ['SDSS-RM','XMM-LSS','COSMOS','AQMES-Medium','AQMES-Wide']#,'eFEDS1','eFEDS2']

### get source info from spAll
### TODO: make spAll file also remote
### spAll is huge and takes >30s to download, which seems to break requests.get
#spAll_url = 'https://data.sdss5.org/sas/sdsswork/bhm/boss/spectro/redux/v6_0_2/spAll-v6_0_2.fits'
#a = requests.get(spAll_url, auth=(username,password))
#spAll = fits.open(io.BytesIO(a.content)) ## path to spALL-master file
spAll = fits.open('./spAll-v6_0_2.fits')

### 
### the webpage layout 
###
app.layout = html.Div(className='container',children=[
    html.H2(children=['SDSSV-BHM Spectra Viewer (remote version)']),

    html.Div([

        ## dropdown menu titles
        html.Div([
            html.H4(children=['Program'])
        ],style={"width": "33%",'display': 'inline-block'}),

        ## plate ID dropdown
        html.Div(children=[
             html.H4(children=['Plate ID'])
        ],style={"width": "33%",'display': 'inline-block'}),

        ## catalog ID dropdown
        html.Div(children=[
             html.H4(children=['Catalog ID'])
        ],style={"width": "33%",'display': 'inline-block'}),

    ]),

    html.Div([

        ## dropdown menu for program/designid/catalogid
        html.Div([
        dcc.Dropdown(
            id='program_dropdown',
            options=[
                {'label': i, 'value': i} for i in programs.keys()],
            placeholder="Program",
            value='SDSS-RM',
            #style={'display': 'inline-block'},
        )],style={"width": "33%",'display': 'inline-block'}),

        ## plate ID dropdown
        html.Div(children=[
        dcc.Dropdown(
            id='plateid_dropdown',
            placeholder='Plate ID',
            value='all',
            #style={'width':'50%','display': 'inline-block'},
        )],style={"width": "33%",'display': 'inline-block'}),


        ## catalog ID dropdown
        html.Div(children=[
        dcc.Dropdown(
            id='catalogid_dropdown',
            placeholder='Catalog ID',
            #style={'width':'50%','display': 'inline-block'},
        )],style={"width": "33%",'display': 'inline-block'}),

        html.Table([
        html.Tr([html.Td('RA'),html.Td('DEC'),html.Td('z'),html.Td('Class'),html.Td('Subclass')]),
        html.Tr([html.Td(id='ra'),html.Td(id='dec'),html.Td(id='z'),html.Td(id='mainclass'),html.Td(id='subclass')]),
        ]),

    ]),

    dcc.Store(id='intermediate-value'),

    ## multiepoch spectra plot
    dcc.Checklist(
        id="epoch_list",
        labelStyle={'display': 'inline-block'}
    ),
    dcc.Graph(id="spectra_plot"),

    html.Div([

        ## spectral binning
        html.Div(children=[
            html.H4(children=['Smoothing:'])
        ],style={"width": "10%",'display': 'inline-block'}),

        html.Div(children=[
            dcc.Input(id="binning_input", type="number", value=10),
        ],style={"width": "15%",'display': 'inline-block'}),

    ]),

        html.Div([

        ## label important spectral lines
        
        html.Div(children=[
            html.H4(children=['Lines:'])
        ],style={"width": "10%",'display': 'inline-block'}),   

        html.Div(children=[
            dcc.Checklist(id="line_list",options=[
                {'label': i+' ('+str(int(spectral_lines[i][0]))+'A)', 'value': i} for i in spectral_lines.keys()
                ], 
                value=list(spectral_lines.keys())),
        ],style={"width": "80%", 'display': 'inline-block'}),     

    ]),

    dcc.Graph(id="residual_plot"),


])


###
### interactive callback functions for updating spectral plot
###

## dropdown menu
@app.callback(
    Output('plateid_dropdown', 'options'),
    Input('program_dropdown', 'value'))
def set_plateid_options(selected_program):
    return [{'label': i, 'value': i} for i in programs[selected_program]]

@app.callback(
    Output('catalogid_dropdown', 'options'),
    Input('plateid_dropdown', 'value'),
    Input('program_dropdown', 'value'))
def set_catalogid_options(selected_designid, selected_program):
    if selected_designid != 'all':
        return [{'label': i, 'value': i} for i in plateIDs[str(selected_designid)]]
    else:
        return [{'label': i, 'value': i} for i in plateIDs[str(selected_program) +"-"+str(selected_designid)]]

@app.callback(
    Output('plateid_dropdown', 'value'),
    Input('plateid_dropdown', 'options'))
def set_plateid_value(available_plateid_options):
    return available_plateid_options[0]['value']

@app.callback(
    Output('catalogid_dropdown', 'value'),
    Input('catalogid_dropdown', 'options'))
def set_catalogid_value(available_catalogid_options):
    return available_catalogid_options[0]['value']

## get all the spectra data 
## use DCC store to store data temporarily in browser

@app.callback(
    Output('intermediate-value', 'data'), 
    Input('plateid_dropdown', 'value'),
    Input('catalogid_dropdown', 'value'),)
def clean_data(selected_designid,selected_catalogid):
    # some expensive clean data step
    get_data = fetch_catID(selected_catalogid, selected_designid)
    # more generally, this line would be
    # json.dumps(cleaned_df)
    return get_data.to_json(date_format='iso', orient='split')

## plotting the spectra
@app.callback(
    Output('spectra_plot','figure'),
    Input('intermediate-value','data'),
    Input('catalogid_dropdown', 'value'),
    Input('binning_input', 'value'),
    Input('line_list', 'value'))
def make_multiepoch_spectra(multiepoch_spectra, selected_catalogid, binning, plot_lines):
    df = pd.read_json(multiepoch_spectra, orient='split')
    flux_limit = 0.
    epochs = np.array(df.index)

    fig = go.Figure()

    for i in np.array(df.index):
        wave_ma = np.convolve(np.array(df.columns), np.ones(binning), 'valid') / binning ## moving average
        flux_ma = np.convolve(df.loc[i], np.ones(binning), 'valid') / binning
        ind = np.where(np.all([wave_ma<wave_max,wave_ma>wave_min],axis=0))
        if np.max(flux_ma[ind])>flux_limit: flux_limit = np.max(flux_ma[ind])
        fig.add_trace(go.Scatter(x=wave_ma[ind], y=flux_ma[ind], name=int(i), \
                                 opacity = 0.3, mode='lines', \
                                 line=dict(color=get_continuous_color(colorscale, \
                                 intermed=(i-np.min(np.array(df.index)))/(np.max(np.array(df.index))-np.min(np.array(df.index))))), \
                                 ))

    z_obj = np.median(spAll[1].data['z'][np.where(spAll[1].data['catalogid']==selected_catalogid)[0]])
    for l in plot_lines:
        for ll in spectral_lines[l]:
            if ll*(1.+z_obj) > np.max([wave_min,np.min(np.array(df.columns))]) and ll*(1.+z_obj) < np.min([wave_max,np.max(np.array(df.columns))]): 
                fig.add_vline(x=ll*(1.+z_obj),line_width=1.5,line=dict(dash='dot'),opacity=0.5)
                fig.add_annotation(x=ll*(1.+z_obj), y=flux_limit,text=l,showarrow=False,xshift=10)
    
    fig.update_layout(xaxis = dict(tickmode = 'linear', tick0 = wave_min, dtick = 1000), \
                      xaxis_tickformat = 'd', yaxis_tickformat = 'd', \
                      xaxis_title="Wavelength (A)", yaxis_title="Flux (1e-17 erg/cm2/s/A)", legend_title="MJD", \
                      )

    return fig

@app.callback(
    Output('residual_plot','figure'),
    Input('intermediate-value','data'),
    Input('catalogid_dropdown', 'value'),
    Input('binning_input', 'value'),
    Input('line_list', 'value'))
def make_multiepoch_spectra(multiepoch_spectra, selected_catalogid , binning, plot_lines):
    df = pd.read_json(multiepoch_spectra, orient='split')
    epochs = np.array(df.index)

    allspec = df.values
    for i in range(allspec.shape[0]):
        allspec[i,:] = (allspec[i,:]-np.median(allspec,axis=0))#/np.median(allspec,axis=0)
    #fig = go.Figure()
    fig = px.imshow(allspec,x=np.array(df.columns),y=np.array(df.index),aspect='auto', \
        color_continuous_midpoint=0, color_continuous_scale=px.colors.diverging.PRGn)

    
    z_obj = np.median(spAll[1].data['z'][np.where(spAll[1].data['catalogid']==selected_catalogid)[0]])
    for l in plot_lines:
        for ll in spectral_lines[l]:
            if ll*(1.+z_obj) > np.max([wave_min,np.min(np.array(df.columns))]) and ll*(1.+z_obj) < np.min([wave_max,np.max(np.array(df.columns))]): 
                fig.add_vline(x=ll*(1.+z_obj),line_width=1.5,line=dict(dash='dot'),opacity=0.5)
                fig.add_annotation(x=ll*(1.+z_obj), y=np.min(epochs)+5.,text=l,showarrow=False,xshift=10)
    
    fig.update_layout(xaxis = dict(tickmode = 'linear', tick0 = wave_min, dtick = 1000), \
                      xaxis_tickformat = 'd', yaxis_tickformat = 'd', \
                      xaxis_title="Wavelength (A)", yaxis_title="Epoch(MJD)")

    return fig

## calling source info from spAll
## TO DO: change to calling info from dictionaries.txt
@app.callback(
    Output('ra', 'children'),
    Output('dec', 'children'),
    Output('z', 'children'),
    Output('mainclass', 'children'),
    Output('subclass', 'children'),
    #Output('Simbad','children'),
    Input('catalogid_dropdown', 'value'))
def source_info(selected_catalogid):
    ra_deg = '%.6f' % spAll[1].data['plug_ra'][np.where(spAll[1].data['catalogid']==selected_catalogid)[0]][0]
    dec_deg = '%.6f' % spAll[1].data['plug_dec'][np.where(spAll[1].data['catalogid']==selected_catalogid)[0]][0]
    #coord = SkyCoord(ra_deg,dec_deg,unit='deg')
    #ra = coord.ra.to_string(u.hour)
    #dec = coord.dec.to_string(u.deg)
    z = '%.4f' % np.median(spAll[1].data['z'][np.where(spAll[1].data['catalogid']==selected_catalogid)[0]])
    cl = spAll[1].data['class'][np.where(spAll[1].data['catalogid']==selected_catalogid)[0]]
    scl = spAll[1].data['subclass'][np.where(spAll[1].data['catalogid']==selected_catalogid)[0]]
    mc_values, mc_counts = np.unique(cl[np.where(cl!=' ')], return_counts=True)
    sc_values, sc_counts = np.unique(scl[np.where(scl!=' ')], return_counts=True)
    if len(mc_counts)==1:
        mainclass = mc_values[np.argmax(mc_counts)]
    else:
        mainclass = mc_values[np.argmax(mc_counts)]+' ('+mc_values[np.where(np.argsort(mc_counts)==len(mc_counts)-2)[0]]+')'
    if len(sc_counts)==1:
        subclass = sc_values[np.argmax(sc_counts)]
    else:
        subclass = sc_values[np.argmax(sc_counts)]+' ('+sc_values[np.where(np.argsort(sc_counts)==len(sc_counts)-2)[0]]+')'

    return ra_deg, dec_deg, z, mainclass, subclass#, simbad_link

### setting the selected epochs for plotting
# @app.callback(
#     Output('epoch_list','value'),
#     Input('plateid_dropdown', 'value'),
#     Input('catalogid_dropdown', 'value'))
# def set_epoch_value(selected_designid,selected_catalogid):
#     filename = np.array([])
#     for i in plateid[selected_designid]:
#         tmp = glob.glob(dir_spectra+str(i)+'p/coadd/*/spSpec-'+str(i)+'-*-'+str(selected_catalogid).zfill(11)+'.fits')
#         if len(tmp)>0: 
#             filename = np.append(filename,tmp,axis=0)
#     epoch = np.array([])
#     for f in filename:
#         mjd = f.split('/')[-2]
#         plate = f.split('/')[-4][:5]
#         epoch = np.append(epoch,float(plate)+float(mjd)/1e5)
#     return [{'label':i, 'value':i} for i in epoch]

if __name__ == '__main__':
    app.run_server(debug=True,host="localhost",port=8054)


