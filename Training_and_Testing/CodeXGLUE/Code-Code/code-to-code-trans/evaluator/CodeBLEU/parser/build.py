from tree_sitter import Language

Language.build_library(
    'my-languages.so',
    [
        'tree-sitter-java'
    ]
)

