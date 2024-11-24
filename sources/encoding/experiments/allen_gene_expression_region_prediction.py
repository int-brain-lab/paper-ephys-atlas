import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from iblatlas.atlas.genes import allen_gene_expression
from iblatlas.atlas import AllenAtlas

ba = AllenAtlas()
df_genes, gene_expression = allen_gene_expression()

ne, nml, ndv, nap = DIM_EXP = gene_expression.shape
ccf_coords = np.meshgrid(
    *[np.arange(DIM_EXP[i]) * 200 for i in [1, 2, 3]]
)  # ml, dv, ap
xyzs = ba.ccf2xyz(
    np.c_[ccf_coords[0].flatten(), ccf_coords[2].flatten(), ccf_coords[1].flatten()]
)
aids = ba.get_labels(xyzs, mode="clip", mapping="Cosmos")

sel = ~np.logical_or(aids == 0, aids == 997)  # remove void and root voxels
# reshape in a big array nexp x nvoxels
gexps = (
    gene_expression.reshape((ne, nml * nap * ndv))[:, sel]
    .astype(np.float32)
    .transpose()
)
xyzs = xyzs[sel]
aids = aids[sel]

# train a basic classifier
X_train, X_test, y_train, y_test = train_test_split(gexps, aids, stratify=aids)
scaler = StandardScaler()
scaler.fit(gexps)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


clf = MLPClassifier(random_state=1, max_iter=300, verbose=True).fit(X_train, y_train)
clf.predict_proba(X_test[:1])

clf.predict(X_test)

clf.score(X_test, y_test)
