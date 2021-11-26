import numpy as np
import pandas as pd
import streamlit as st
from plotnine import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import glob
import datetime

path = r'C:\Users\Fabrício\Documents\NuBank' 
all_files = glob.glob(path + '/nubank*.csv')

li = []

for filename in all_files:
    base = pd.read_csv(filename, index_col=None, header=0,error_bad_lines=False)
    li.append(base)


# ganho, gasto e saldo
# ganho por categoria
# gasto por categoria
# ganho mes
# gasto mes
# Balance mes
# representa qual percentual do meu ganho o gasto por categoria
# representa qual percentual do meu ganho o gasto por categoria por mês
# saldo líquido mês a mês

pd.set_option('display.precision', 2)

def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

df = pd.read_csv('DF_NUBANK_TRAIN.csv')

df_credito = pd.concat(li, axis=0, ignore_index=True)

df_credito['category'] = np.where(df_credito['category'] == 'viagem', 'aluguel',df_credito['category'])

df_credito['category'] = np.where(df_credito['category'] == 'outros', 'serviços',df_credito['category'])


df_credito['amount'] =df_credito['amount'].apply(lambda x: -1*x)

df_credito['date'] = pd.to_datetime(df_credito['date'],format='%d/%m/%Y',infer_datetime_format=True)
df['Data'] = pd.to_datetime(df['Data'],format='%d/%m/%Y',infer_datetime_format=True)

df_credito.rename(columns = {'date':'Data',
                    'amount' : 'Valor',
                    'title' : 'Descrição'},
                    inplace = True)
df_credito = df_credito[['Data','Valor','Descrição','category']].dropna()

df = df.append(df_credito)

df = df[df['Descrição'].str.match('^(?!Pagamento da fatura - Ca).*$')]

df['Data'] = pd.to_datetime(df['Data'],format='%d/%m/%Y')

df['Valor'] = df['Valor'].apply(lambda x: truncate(x,0))

start_date = st.date_input('Start date',value=df["Data"].min(), min_value = df["Data"].min())
end_date = st.date_input('End date', value=df["Data"].min(),max_value = df["Data"].max())

st.markdown(
    """"""
    f'Conta periodo de: {start_date} a {end_date}'
    """"""
    )

df = df.query(f'Data >= "{start_date}" and Data <= "{end_date}"')

df['month'] = df['Data'].dt.month

ganho = df.query("category == 'trabalho' | category == 'cursos'").agg(soma = ("Valor", "sum")).reset_index(drop=['index'])['Valor'].sum()

gasto = df.query("category != 'trabalho' & category != 'cursos'").agg(soma = ("Valor", "sum")).reset_index(drop=['index'])['Valor'].sum()

col1, col2, col3 = st.columns(3)

col1.metric('Ganho no período:', truncate(ganho,0))

col2.metric('Gasto no período:', truncate(gasto,0))

col3.metric('Saldo no período:', truncate(df["Valor"].sum(),2))


df_mes = df.groupby("month").agg(Soma = ("Valor","sum")).reset_index()
df_mes['Soma'] = df_mes['Soma'].apply(lambda x: truncate(x,0))

df_mes['pos'] = np.where(df_mes['Soma']>0, 'positive','negative')


df_mes['month'] = pd.Categorical(df_mes['month'])


p = (
    ggplot(df_mes, aes(x ='month' ,y='Soma', fill = 'pos')) +
    geom_bar(stat = 'identity') +
    geom_text(aes(label = 'Soma', va=np.where(df_mes['Soma']>0,'bottom','top')), size=7)+
    scale_fill_manual(values = ["red","blue"],guide=False)+
    labs(x='Meses',
         y = 'Saldo')+
    theme_bw()
)

st.pyplot(ggplot.draw(p))




# aqui somar por categoria acaba anulando o que foi estornado
df_categoria = df.groupby(['category']).agg(Soma = ('Valor','sum')).reset_index()

df_categoria['pos'] = np.where(df_categoria['Soma']>0, 'positive','negative')

df_categoria_pos = df_categoria.query('pos == "positive"')

df_categoria_pos['Soma'] = df_categoria_pos['Soma'].apply(lambda x: truncate(x,0))

df_categoria_pos['Perc'] = 100*(df_categoria_pos['Soma']/df_categoria['Soma'].sum())
df_categoria_pos['Perc'] = df_categoria_pos['Perc'].apply(lambda x: truncate(x,0))

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct
# Data to plot

labels = df_categoria_pos['category']
sizes = df_categoria_pos['Soma']
explode = (0.1, 0.1)  # explode 1st slice

# Plot
fig = plt.figure(figsize=(5, 3))
ax = fig.add_subplot()

ax.pie(sizes, explode=explode, labels=labels, autopct=make_autopct(sizes), startangle=140,textprops={'fontsize':4})

st.pyplot(fig)



df_categoria_neg = df_categoria.query('pos == "negative"')


df_categoria_neg['Soma'] = df_categoria_neg['Soma'].apply(lambda x: truncate(x,0))

df_categoria_neg['Perc'] = 100*(df_categoria_neg['Soma']/df_categoria['Soma'].sum())
df_categoria_neg['Perc'] = df_categoria_neg['Perc'].apply(lambda x: truncate(x,0))


labels = df_categoria_neg['category']
sizes = -1*df_categoria_neg['Soma']
#explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)  # explode 1st slice

# Plot
fig = plt.figure(figsize=(5, 3))
ax = fig.add_subplot()

ax.pie(sizes,  labels=labels, autopct=make_autopct(sizes), startangle=140,textprops={'fontsize':4})

st.pyplot(fig)


df_mes_ganho = df.query("category == 'trabalho' | category == 'cursos'").groupby(["month"]).agg(Soma = ("Valor","sum")).reset_index()
df_mes_ganho['Soma'] = df_mes_ganho['Soma'].apply(lambda x: truncate(x,0))
df_mes_ganho['month'] = pd.Categorical(df_mes_ganho['month'])


p = (
    ggplot(df_mes_ganho, aes(x ='month' ,y='Soma')) +
    geom_bar(stat = 'identity', fill = 'blue') +
    geom_text(aes(label = 'Soma' ),va='bottom', size=7)+
    labs(x='Meses',
         y = 'Ganhos')+ 
    theme_bw()
)

st.pyplot(ggplot.draw(p))


df_mes_ganho_category = df.query("category == 'trabalho' | category == 'cursos'").groupby(["month","category"]).agg(Soma = ("Valor","sum")).reset_index()
df_mes_ganho_category['Soma'] = df_mes_ganho_category['Soma'].apply(lambda x: truncate(x,0))
df_mes_ganho_category['month'] = pd.Categorical(df_mes_ganho_category['month'])



p = (
    ggplot(df_mes_ganho_category, aes(x ='category' ,y='Soma', fill = 'category')) +
    geom_bar(stat = 'identity') +
    geom_text(aes(label = 'Soma' ),va='bottom', size=7)+
    labs(x='Meses',
         y = 'Ganhos')+ 
    lims(y = [0,40000]) +
    scale_fill_manual(values = ["red","blue"],guide=False)+
    facet_wrap('month', ncol = 1) +
    theme_bw() +
    theme(figure_size=(6, 16)) 
)

st.pyplot(ggplot.draw(p))

df_mes_perda = df.query("category != 'trabalho' & category != 'cursos'").groupby(["month"]).agg(Soma = ("Valor","sum")).reset_index()
df_mes_perda['Soma'] = df_mes_perda['Soma'].apply(lambda x: truncate(x,0))
df_mes_perda['month'] = pd.Categorical(df_mes_perda['month'])

p = (
    ggplot(df_mes_perda, aes(x ='month' ,y='Soma')) +
    geom_bar(stat = 'identity', fill = 'red') +
    geom_text(aes(label = 'Soma' ),va='top', size=7)+
    labs(x='Meses',
         y = 'Gastos')+
    theme_bw()
)

st.pyplot(ggplot.draw(p))



df_mes_perda_category = df.query("category != 'trabalho' & category != 'cursos'").groupby(["month","category"]).agg(Soma = ("Valor","sum")).reset_index()
df_mes_perda_category['Soma'] = df_mes_perda_category['Soma'].apply(lambda x: truncate(x,0))
df_mes_perda_category['month'] = pd.Categorical(df_mes_perda_category['month'])

p = (
    ggplot(df_mes_perda_category, aes(x ='category' ,y='Soma', fill = 'category')) +
    geom_bar(stat = 'identity') +
    geom_text(aes(label = 'Soma' ),va='top', size=5)+
    labs(x='Meses',
         y = 'Ganhos')+ 
    lims(y = [-20000,10000]) +
    scale_fill_discrete(guide=False)+
    facet_wrap('month', ncol = 1) +
    theme_bw() +
    theme(axis_text_x=element_text(rotation=90, hjust=1))+
    theme(figure_size=(6, 16))    
)

st.pyplot(ggplot.draw(p))

df_IR = pd.read_csv('declaracao-de-movimentacao-bancaria.csv')

df_IR['Data'] = pd.to_datetime(df_IR['Data'])

df_IR['month'] = df_IR['Data'].dt.month

saldo_IR = df_IR['Saldo Líquido'].tail(1)

rendimento_IR = df_IR.agg(soma = ("Rendimentos", "sum")).reset_index(drop=['index'])['Rendimentos'].sum()

col1, col2 = st.columns(2)

col1.metric('Último saldo:', truncate(saldo_IR,0))

col2.metric('Rendimentos no período:', truncate(rendimento_IR,0))


saldo_mes = df_IR.groupby(['month']).agg(media_saldo = ("Saldo Líquido", "mean")).reset_index()
saldo_mes['media_saldo'] = saldo_mes['media_saldo'].apply(lambda x :truncate(x,0))

p = (
    ggplot(saldo_mes, aes(x ='month' ,y='media_saldo')) +
    geom_bar(stat = 'identity', fill = 'green') +
    geom_text(aes(label = 'media_saldo'), va = 'bottom',size=6) +
    labs(x='Meses',
         y = 'Ganhos')+ 
    theme_bw() +
    theme(axis_text_x=element_text(rotation=90, hjust=1))    
)

st.pyplot(ggplot.draw(p))


rendimentos_mes = df_IR.groupby(['month']).agg(Rendimentos = ('Rendimentos','sum')).reset_index()

p = (
    ggplot(rendimentos_mes, aes(x ='month' ,y='Rendimentos')) +
    geom_bar(stat = 'identity', fill = 'yellow') +
    geom_text(aes(label = 'Rendimentos'), va = 'bottom', size = 6) +
    labs(x='Meses',
         y = 'Rendimentos')+ 
    theme_bw() +
    theme(axis_text_x=element_text(rotation=90, hjust=1))    
)

st.pyplot(ggplot.draw(p))
