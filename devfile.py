from nrclex import NRCLex

# import pandas as pd
# import conf

# datadf = pd.read_csv(conf.DATAPATH)

aa = NRCLex('I hate everything about you')
bb = NRCLex("I love you so much")
# aa.affect_frequencies
print(NRCLex('I hate everything about you').affect_frequencies)
print("pass")
