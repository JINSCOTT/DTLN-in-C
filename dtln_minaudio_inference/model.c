#include "model.h"


struct model* create_model() {
	struct model* NewModel = (struct model*)malloc(sizeof(struct model));
	if (NewModel) {
		NewModel->node.first = NULL;
		NewModel->node.size = 0;
		NewModel->tensor.first = NULL;
		NewModel->tensor.size = 0;
		printf("model allocated. \n");
	}
	return NewModel;
}

//  copied into, free to edlete
int16_t set_model_tensor_data(struct model* m, int64_t index, void* data, int64_t datasize){
	struct tensor* rTensor = (struct tensor*)get_data_list(&m->tensor, index);
	if (rTensor == NULL) return 0;
	memcpy(rTensor->data, data, datasize);
	return 1;

}

// Do not free data got
int16_t get_model_tensor_data(struct model* m, int64_t index, void** data) {
	struct tensor* rTensor = (struct tensor*)get_data_list(&m->tensor, index);
	if (rTensor == NULL) return 0;
	*data = rTensor->data;
	return 1;
}

int inference_model(struct model* m) {
	int num_nodes = 0, error = 0, i = 0;
	struct node* cur_node = NULL;
	num_nodes = m->node.size;
	for (i = 0; i < num_nodes; i++) {
		cur_node = (struct Node*)get_data_list(&m->node, i);
		if (cur_node == NULL) {
			printf("Encountered NULL node\n");
			return 0;
		}
		error = inference_node(cur_node);
		if (error!= 1) {
			printf("Node execution failed\n");
			return 0;
		}
	}
	return 1;
}

#ifdef DEBUG
void print_model(struct model* model) {
	//if (model->Name != NULL) printf("%s:\n", model->Name);
	return;
};
#endif // DEBUG