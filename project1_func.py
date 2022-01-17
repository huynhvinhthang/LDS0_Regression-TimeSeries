#import library
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#create season feature
def season(m):
    if m == 3 or m ==4 or m ==5:
        return 0#spring
    elif m == 6 or m==7 or m==8:
        return 1#summer
    elif m==9 or m ==10 or m==11:
        return 2 #fall
    else:
        return 3#winter

#anova
def anova(f1, f2, f3, data):
    print('-'*20, 'Anova between {}, {} and {}'.format(f1,f2,f3), '-'*20)
    model = ols('{} ~ C({}) + C({}) + C({}):C({})'.format(f1,f2,f3,f2,f3), data = data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

#Tukey_hsd test
def tukeyhsd(f1, f2, f3, data):
    print('-'*20, 'TukeyHSD test between {}, {} and {}'.format(f1,f2,f3), '-'*20)
    print('\n')
    for name, grouped_df in data.groupby(f2):
        print('{} {}'.format(f2, name), pairwise_tukeyhsd(grouped_df[f1],
                                                         grouped_df[f3], alpha = 0.05))

from scipy.stats import chi2_contingency, chi2
#chi2
def two_categorical(table):
  stat, p, dof, expected = chi2_contingency(table)
  print('dof = %d' % dof)
  #print(expected)
  print('==============Interpret test-statistic===============')
  prob = 0.95
  critical = chi2.ppf(prob, dof)
  print('Probability = %.3f, Criticcal = %.3f, Stat = %.3f' %(prob, critical, stat))
  if abs(stat) >= critical:
    print('Dependent (Reject H0)')
  else:
    print('Independent (Fail to reject H0)')
  print('==============Interpret P-value===============')
  alpha = 1-prob
  print('Significance = %.3f, p = %.3f' %(alpha, p))
  if p<alpha:
    print('Dependent (Reject H0)')
  else:
    print('Independent (Fail to reject H0)')


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
def season_decompose(data, feature):
    result = seasonal_decompose(data[feature], model = 'multiplicative', period = 52)
    fig, v8 = plt.subplots()
    #fig.rcParams['figure.figsize'] = (14, 9)
    result.plot(ax = v8)
    plt.show()