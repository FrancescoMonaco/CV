#ifndef Constants
#define Constants
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>

const int bigRadius = 295;
const int smallRadius = 287;
const double lowerBound = 0.1;
const double BREAD_THRESH = 0.73;
const double SALAD_THRESH = 0.42;

const std::string PREDS_BB = "/box_preds/";
const std::string PREDS_SS = "/mask_preds/";

const int BACKGROUND = 0, PESTO = 1, TOMATO = 2, MEAT_SAU = 3, CLAMS = 4, RICE = 5,\
PORK = 6, FISH = 7, RABBIT = 8, SEAFOOD = 9, BEANS = 10, POTATOES = 11, SALAD = 12, BREAD = 13;

#endif // !Constants
