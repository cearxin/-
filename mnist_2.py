plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i].reshape(28,28), cmap=plt.cm.binary)
plt.show()

# + prediction

# predictions = model.predict(x_test[:25])
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_test[i].reshape(28,28), cmap=plt.cm.binary)
#     plt.xlabel(f"Predicted: {predictions.argmax(axis=1)[i]}")
# plt.show()
