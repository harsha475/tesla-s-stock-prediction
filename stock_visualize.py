# For visualization of precision score
import matplatlib.pyplot as plt
feature_importances = pd.Series(best_model.feature_importances_, index=predictors).sort_values(ascending=False)
feature_importances.plot(kind="bar", title="Feature Importance")
plt.show()
