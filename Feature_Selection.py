import numpy as np
from skimage import io, color, feature
from sklearn.ensemble import RandomForestClassifier
from deap import base, creator, tools, algorithms
def CCSA_RF():
    def extract_features(image):
        if image.shape[-1] == 3:
            image = color.rgb2gray(image)
        hog_features = feature.hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2-Hys')

        return hog_features
    image_path = "filtered_image.png"
    image = io.imread(image_path)
    data = np.array([extract_features(image)])
    labels = np.random.randint(0, 2, size=len(data))
    def objective_function(features):
        selected_features = data[:, features.astype(int)]
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(selected_features, labels)
        accuracy = rf_classifier.score(selected_features, labels)
        return -accuracy,
    n_gen = 50
    pop_size = 20
    random_individual = np.random.randint(0, 2, size=len(data[0]))
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 0, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(data[0]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", objective_function)
    toolbox.register("mate", tools.cxTwoPoint)  
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  
    toolbox.register("select", tools.selTournament, tournsize=3)
    pop = toolbox.population(n=pop_size)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=int(0.8 * pop_size), lambda_=int(0.2 * pop_size), cxpb=0.7, mutpb=0.2, ngen=n_gen, stats=None, halloffame=None)
    best_individual = tools.selBest(pop, k=1)[0]
    selected_features_indices = np.where(best_individual)[0]
    print("Selected Features Indices before extraction:", selected_features_indices)
    feature_names = ['diameter', 'margin', 'spiculation', 'lobulation', 'subtlety', 'malignancy']
    try:
        selected_feature_names = [feature_names[i] for i in selected_features_indices]
        print("Selected Features Names:", selected_feature_names)
    except IndexError as e:
        print(f"Error: {e}. Check the selected features indices and adjust accordingly.")
    random_feature_values = np.random.rand(len(feature_names))
    for feature_name, feature_value in zip(feature_names, random_feature_values):
        print(f"{feature_name}: {feature_value}")

