import matplotlib.pyplot as plt


style_lambda = [2,4,6,8]
small_textsettr_bleu = [0.6961333800414901, 0.5220095500939619, 0.31344046007212173, 0.1859248527440013]
small_textsettr_pred = [0.016655562958027982, 0.14054484785098859, 0.47800135376284536, 0.7048630519843488]
textsettr_bleu = [0.530495318833397, 0.4807084553348837, 0.3955154511404267, 0.3136000090082577]
textsettr_pred = [0.03561892046538241, 0.05815339148672482, 0.0925589836660617, 0.14425016812373906]
merged_textsettr_bleu = [0.73391013173726621, 0.55915919570210693, 0.28891754485971410, 0.161076446102383174]
merged_textsettr_pred = [0.01054103968632295, 0.13810677669544428, 0.49498108728201640, 0.7139667449469419]

fig, left = plt.subplots()
right = left.twinx()
right.spines["right"].set_visible(True)
left.set_ylabel('BLEU score')
right.set_ylabel('Classifier score')

plt.xticks(style_lambda)
left.set_xlabel('Style lambda')

left.plot(style_lambda, small_textsettr_bleu, color='green', marker='x', label='Small TextSETTR (BELU)')
right.plot(style_lambda, small_textsettr_pred, color='lime', marker='x', label='Small TextSETTR (Classifier)')

left.plot(style_lambda, textsettr_bleu, color='deepskyblue', marker='x', label='TextSETTR (BELU)')
right.plot(style_lambda, textsettr_pred, color='skyblue', marker='x', label='TextSETTR (Classifier)')

left.plot(style_lambda, merged_textsettr_bleu, color='darkred', marker='x', label='Merged TextSETTR (BELU)')
right.plot(style_lambda, merged_textsettr_pred, color='red', marker='x', label='Merged TextSETTR (Classifier)')

left.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2))
right.legend(loc='lower right', bbox_to_anchor=(0, 1.02, 1, 0.2))
plt.show()
