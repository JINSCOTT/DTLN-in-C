#include "model.h"


struct model* create_model() {
	struct model* NewModel = (struct model*)malloc(sizeof(struct model));
	if (NewModel) {
		NewModel->node.first = NULL;
		NewModel->node.size = 0;
		NewModel->tensor.first = NULL;
		NewModel->tensor.size = 0;
	}
	return NewModel;
}

int inference_model(struct model* m) {
	int  error = 0;
	int64_t num_nodes = 0, i = 0;
	struct Node* cur_node = NULL;
	num_nodes = m->node.size;
	for (i = 0; i < num_nodes; i++) {
		printf ("%d", i);
		cur_node = (struct Node*)get_list(&m->node, i);
		if (cur_node == NULL) {
			printf("Encountered NULL node\n");
			return 0;
		}
		error = inference_node(cur_node);
		if (error!= 1) {
			printf("Node execution failed\n");
			return 0;
		}
		//system("pause");
	}
	return 1;
}

#ifdef DEBUG
void print_model(struct model* model) {
	//if (model->Name != NULL) printf("%s:\n", model->Name);
	return;
};
#endif // DEBUG