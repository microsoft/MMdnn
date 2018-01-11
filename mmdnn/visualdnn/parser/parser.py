 sequence of statements to translate:

def fib(n):
    if n < 2:
        return 1
    return fib(n - 1) + fib(n - 2)

print fib(20)

# ------------------------------------------------------------------------------------------------

# AST nodes:

class Node(object):
    def __repr__(self):
        if hasattr(self.__init__, 'im_func'):
            names = self.__init__.im_func.func_code.co_varnames[1:]
            s = ", ".join(repr(getattr(self, name)) for name in names)
        else: s = ""
        return "%s(%s)" % (self.__class__.__name__, s)

class Stmt(Node):
    pass

class Expr(Node):
    pass

class Suite(Stmt):
    def __init__(self, *stmts):
        self.stmts = stmts

class Func(Stmt):
    def __init__(self, name, params, suite):
        self.name, self.params, self.suite = name, params, suite

class If(Stmt):
    def __init__(self, cond, tsuite, esuite):
        self.cond, self.tsuite, self.esuite = cond, tsuite, esuite

class Pass(Stmt):
    pass

class Return(Stmt):
    def __init__(self, expr):
        self.expr = expr

class Println(Stmt):
    def __init__(self, expr):
        self.expr = expr

class Var(Expr):
    def __init__(self, name):
        self.name = name

class Lit(Expr):
    def __init__(self, value):
        self.value = value

class BinOp(Expr):
    def __init__(self, left, right):
        self.left, self.right = left, right

class Lt(BinOp):
    pass

class Add(BinOp):
    pass

class Sub(BinOp):
    pass

class Call(Expr):
    def __init__(self, func, *exprs):
        self.func, self.exprs = func, exprs

# translated statement sequence:

suite = Suite(
    Func("fib", ["n"], Suite(
        If(Lt(Var("n"), Lit(2)), Suite(Return(Lit(1))), Pass()),
        Return(Add(
            Call(Var("fib"), Sub(Var("n"), Lit(1))), 
            Call(Var("fib"), Sub(Var("n"), Lit(2))))))),
    Println(Call(Var("fib"), Lit(20))))

print suite

# ------------------------------------------------------------------------------------------------

# runtime representation of Python values

class PyValue:
    def pynonzero(self, frame):
        return False
    
    def pystr(self, frame):
        return PyStr("<%s object>" % self.__class__.__name__[2:])
    
    def pycall(self, frame, args):
        raise Exception("not callable")

class PyNone(PyValue):
    def pynonzero(self, frame):
        return True
    
    def pystr(self, frame):
        return PyStr("None")

class PyInt(PyValue):
    def __init__(self, value):
        self.value = value
    
    def pynonzero(self, frame):
        return bool(self.value)
    
    def pystr(self, frame):
        return PyStr(str(self.value))
    
    def pylt(self, other):
        if isinstance(other, PyInt):
            return pytrue if self.value < other.value else pyfalse
        return other.pyge(self)
    
    def pyadd(self, other):
        if isinstance(other, PyInt):
            return PyInt(self.value + other.value)
        return other.pyradd(self)
    
    def pysub(self, other):
        if isinstance(other, PyInt):
            return PyInt(self.value - other.value)
        return other.pyrsub(self)

class PyBool(PyInt):
    def pynonzero(self, frame):
        return self == pytrue

    def pystr(self, frame):
        return PyStr("True" if self == pytrue else "False")

class PyStr(PyValue):
    def __init__(self, value):
        self.value = value
    
    def pynonzero(self, frame):
        return bool(self.value)
    
    def pystr(self, frame):
        return self

class PyFunc(PyValue):
    def __init__(self, name, params, suite, globls):
        self.name, self.params, self.suite, self.globls = name, params, suite, globls
    
    def pycall(self, frame, args):
        frame = Frame(frame, self.globls, dict(zip(self.params, args)))
        try:
            self.suite.execute(frame)
        except ReturnException, e:
            return e.result
        return pynone

# well known constants

pynone = PyNone()
pytrue = PyBool(1)
pyfalse = PyBool(0)

# representation of state

class Frame(object):
    def __init__(self, back=None, globls=None, locls=None):
        if locls is None: locls = {}
        if globls is None: globls = locls
        self.back, self.globls, self.locls = back, globls, locls
    
    def getlocal(self, name):
        try:
            return self.locls[name]
        except KeyError:
            return self.getglobal(name)
    
    def setlocal(self, name, value):
        self.locls[name] = value
    
    def dellocal(self, name):
        del self.locls[name]
    
    def getglobal(self, name):
        try:
            return self.globls[name]
        except KeyError:
            try:
                module_or_dict = self.globls["__builtins__"]
                if isinstance(module_or_dict, type(__builtins__)):
                    module_or_dict = module_or_dict.__dict__
                return module_or_dict[name]
            except KeyError:
                raise NameError, name
    
    def setglobals(self, name, value):
        self.globls[name] = value
    
    def delglobals(self, name):
        del self.globls[name]

class ReturnException(Exception):
    def __init__(self, result):
        self.result = result

# ------------------------------------------------------------------------------------------------

# AST evaluator

class Execute:
    def Suite_execute(self, frame):
        for stmt in self.stmts:
            stmt.execute(frame)
    
    def Func_execute(self, frame):
        frame.setlocal(self.name, PyFunc(self.name, self.params, self.suite, frame.globls))

    def If_execute(self, frame):
        (self.tsuite if self.cond.execute(frame).pynonzero(frame) else self.esuite).execute(frame)

    def Pass_execute(self, frame):
        pass
    
    def Return_execute(self, frame):
        raise ReturnException(self.expr.execute(frame))
    
    def Println_execute(self, frame):
        value = self.expr.execute(frame)
        print value.pystr(frame).value
    
    def Var_execute(self, frame):
        return frame.getlocal(self.name)
    
    def Lit_execute(self, frame):
        return PyInt(self.value) # AST should contain PyValues not Python objects
    
    def Lt_execute(self, frame):
        return self.left.execute(frame).pylt(self.right.execute(frame))

    def Add_execute(self, frame):
        return self.left.execute(frame).pyadd(self.right.execute(frame))

    def Sub_execute(self, frame):
        return self.left.execute(frame).pysub(self.right.execute(frame))

    def Call_execute(self, frame):
        return self.func.execute(frame).pycall(frame, (expr.execute(frame) for expr in self.exprs))

for k, v in Execute.__dict__.items():
    if k[0] != '_':
        c, n = k.split("_", 1)
        setattr(globals()[c], n, v)

suite.execute(Frame())