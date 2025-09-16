## How to run tests
test field:
```
g++ -std=c++17 -O2 -Wall -Wextra -I../core test_fields.cpp -o test_fields
```

```
OPTIONS:
  -h,     --help              Print this help message and exit
  -n,     --field INT:{2,3}   Field modulus (2 or 3)
  -a,     --all               Run tests for both B<2> and B<3>
```

example:
```
g++ test_fields.cpp -I../core -std=c++17 -O2 -o test_fields

./test_fields -n 2

./test_fields -n 3

./test_fields --all

# to see doc
./test_fields --help
```
test flip:
```
g++ -std=c++17 -O2 -Wall -Wextra -I../core test_scheme_flip.cpp -o test_scheme_flip
```

```
OPTIONS:
  -h,     --help              Print this help message and exit
  -n,     --field INT:{2,3}   Field modulus (2 or 3)
  -a,     --all               Run flip tests for both B<2> and B<3>
  -q,     --quick             Run only basic tests (faster)
```