#include <game.hpp>
#include <type.hpp>

enum NodeType
{
  OR_NODE,
  AND_NODE
};

struct PNNode
{
  NodeType type;
  uint64_t proof_number;
  uint64_t disproof_number;
  std::vector<std::unique_ptr<PNNode> > children;
  PNNode* parent;
  Position state;

  PNNode (NodeType t, const Position& s, PNNode* p = nullptr)
      : type (t), proof_number (1), disproof_number (1), parent (p), state (s)
  {
  }
};

//// 初始化節點 (根據終局狀態給 proof/disproof)
//void
//initialize_node (PNNode* node)
//{
  //// 檢查終局
  //if ([> state has no legal moves <])
  //{
    //if ([> black win <])
    //{
      //node->proof_number = 0;
      //node->disproof_number = UINT64_MAX;
    //}
    //else
    //{
      //node->proof_number = UINT64_MAX;
      //node->disproof_number = 0;
    //}
  //}
  //else
  //{
    //node->proof_number = 1;
    //node->disproof_number = 1;
  //}
//}

//// 選擇最有希望證明的節點 (最小 proof number 的路徑)

//PNNode*
//select_most_proving_node (PNNode* root)
//{
  //PNNode* node = root;
  //while (!node->children.empty ())
  //{
    //if (node->type == OR_NODE)
    //{
      //// 選 proof_number 最小的
      //node = std::min_element (node->children.begin (),
                               //node->children.end (),
                               //[] (auto& a, auto& b)
                               //{
                                 //return a->proof_number < b->proof_number;
                               //})
                 //->get ();
    //}
    //else
    //{ // AND_NODE
      //// 選 disproof_number 最小的
      //node
          //= std::min_element (node->children.begin (),
                               //node->children.end (),
                               //[] (auto& a, auto& b)
                               //{
                                 //return a->disproof_number < b->disproof_number;
                               //})
                 //->get ();
    //}
  //}
  //return node;
//}

//// 展開一個節點，生成子節點並設置 PN/DN
//void
//expand_node (PNNode* node)
//{
  //// 產生所有合法走法 (你要實作 Othello move generator)
  //std::vector<Position> next_states = generate_moves (node->state);

  //for (auto& s : next_states)
  //{
    //NodeType child_type = (node->type == OR_NODE ? AND_NODE : OR_NODE);
    //node->children.push_back (std::make_unique<PNNode> (child_type, s, node));
    //initialize_node (node->children.back ().get ());
  //}
//}

//// 回傳更新 (自底向上更新 PN/DN)
//void
//update_ancestors (PNNode* node)
//{
  //while (node)
  //{
    //if (node->children.empty ())
    //{
      //// leaf, nothing to update
    //}
    //else if (node->type == OR_NODE)
    //{
      //node->proof_number = UINT64_MAX;
      //node->disproof_number = 0;
      //for (auto& c : node->children)
      //{
        //node->proof_number = std::min (node->proof_number, c->proof_number);
        //node->disproof_number += c->disproof_number;
      //}
    //}
    //else
    //{ // AND_NODE
      //node->proof_number = 0;
      //node->disproof_number = UINT64_MAX;
      //for (auto& c : node->children)
      //{
        //node->proof_number += c->proof_number;
        //node->disproof_number
            //= std::min (node->disproof_number, c->disproof_number);
      //}
    //}
    //node = node->parent;
  //}
//}

//// 主流程：嘗試證明局面
//bool
//proof_number_search (PNNode* root)
//{
  //initialize_node (root);

  //while (root->proof_number != 0 && root->disproof_number != 0)
  //{
    //PNNode* node = select_most_proving_node (root);
    //expand_node (node);
    //update_ancestors (node);
  //}

  //return (root->proof_number == 0); // true = black can force win
//}
