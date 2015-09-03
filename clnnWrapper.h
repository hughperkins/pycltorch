#pragma once

#include <iostream>
#include <string>

class THFloatTensor;
class THFloatStorage;
class THClState;
struct lua_State;

THClState *getState(lua_State *L);

