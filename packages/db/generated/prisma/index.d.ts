
/**
 * Client
**/

import * as runtime from './runtime/library.js';
import $Types = runtime.Types // general types
import $Public = runtime.Types.Public
import $Utils = runtime.Types.Utils
import $Extensions = runtime.Types.Extensions
import $Result = runtime.Types.Result

export type PrismaPromise<T> = $Public.PrismaPromise<T>


/**
 * Model User
 * 
 */
export type User = $Result.DefaultSelection<Prisma.$UserPayload>
/**
 * Model TrainingRun
 * 
 */
export type TrainingRun = $Result.DefaultSelection<Prisma.$TrainingRunPayload>
/**
 * Model Dataset
 * 
 */
export type Dataset = $Result.DefaultSelection<Prisma.$DatasetPayload>

/**
 * ##  Prisma Client ʲˢ
 *
 * Type-safe database client for TypeScript & Node.js
 * @example
 * ```
 * const prisma = new PrismaClient()
 * // Fetch zero or more Users
 * const users = await prisma.user.findMany()
 * ```
 *
 *
 * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client).
 */
export class PrismaClient<
  ClientOptions extends Prisma.PrismaClientOptions = Prisma.PrismaClientOptions,
  const U = 'log' extends keyof ClientOptions ? ClientOptions['log'] extends Array<Prisma.LogLevel | Prisma.LogDefinition> ? Prisma.GetEvents<ClientOptions['log']> : never : never,
  ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs
> {
  [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['other'] }

    /**
   * ##  Prisma Client ʲˢ
   *
   * Type-safe database client for TypeScript & Node.js
   * @example
   * ```
   * const prisma = new PrismaClient()
   * // Fetch zero or more Users
   * const users = await prisma.user.findMany()
   * ```
   *
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client).
   */

  constructor(optionsArg ?: Prisma.Subset<ClientOptions, Prisma.PrismaClientOptions>);
  $on<V extends U>(eventType: V, callback: (event: V extends 'query' ? Prisma.QueryEvent : Prisma.LogEvent) => void): PrismaClient;

  /**
   * Connect with the database
   */
  $connect(): $Utils.JsPromise<void>;

  /**
   * Disconnect from the database
   */
  $disconnect(): $Utils.JsPromise<void>;

/**
   * Executes a prepared raw query and returns the number of affected rows.
   * @example
   * ```
   * const result = await prisma.$executeRaw`UPDATE User SET cool = ${true} WHERE email = ${'user@email.com'};`
   * ```
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/raw-database-access).
   */
  $executeRaw<T = unknown>(query: TemplateStringsArray | Prisma.Sql, ...values: any[]): Prisma.PrismaPromise<number>;

  /**
   * Executes a raw query and returns the number of affected rows.
   * Susceptible to SQL injections, see documentation.
   * @example
   * ```
   * const result = await prisma.$executeRawUnsafe('UPDATE User SET cool = $1 WHERE email = $2 ;', true, 'user@email.com')
   * ```
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/raw-database-access).
   */
  $executeRawUnsafe<T = unknown>(query: string, ...values: any[]): Prisma.PrismaPromise<number>;

  /**
   * Performs a prepared raw query and returns the `SELECT` data.
   * @example
   * ```
   * const result = await prisma.$queryRaw`SELECT * FROM User WHERE id = ${1} OR email = ${'user@email.com'};`
   * ```
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/raw-database-access).
   */
  $queryRaw<T = unknown>(query: TemplateStringsArray | Prisma.Sql, ...values: any[]): Prisma.PrismaPromise<T>;

  /**
   * Performs a raw query and returns the `SELECT` data.
   * Susceptible to SQL injections, see documentation.
   * @example
   * ```
   * const result = await prisma.$queryRawUnsafe('SELECT * FROM User WHERE id = $1 OR email = $2;', 1, 'user@email.com')
   * ```
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/raw-database-access).
   */
  $queryRawUnsafe<T = unknown>(query: string, ...values: any[]): Prisma.PrismaPromise<T>;


  /**
   * Allows the running of a sequence of read/write operations that are guaranteed to either succeed or fail as a whole.
   * @example
   * ```
   * const [george, bob, alice] = await prisma.$transaction([
   *   prisma.user.create({ data: { name: 'George' } }),
   *   prisma.user.create({ data: { name: 'Bob' } }),
   *   prisma.user.create({ data: { name: 'Alice' } }),
   * ])
   * ```
   * 
   * Read more in our [docs](https://www.prisma.io/docs/concepts/components/prisma-client/transactions).
   */
  $transaction<P extends Prisma.PrismaPromise<any>[]>(arg: [...P], options?: { isolationLevel?: Prisma.TransactionIsolationLevel }): $Utils.JsPromise<runtime.Types.Utils.UnwrapTuple<P>>

  $transaction<R>(fn: (prisma: Omit<PrismaClient, runtime.ITXClientDenyList>) => $Utils.JsPromise<R>, options?: { maxWait?: number, timeout?: number, isolationLevel?: Prisma.TransactionIsolationLevel }): $Utils.JsPromise<R>


  $extends: $Extensions.ExtendsHook<"extends", Prisma.TypeMapCb<ClientOptions>, ExtArgs, $Utils.Call<Prisma.TypeMapCb<ClientOptions>, {
    extArgs: ExtArgs
  }>>

      /**
   * `prisma.user`: Exposes CRUD operations for the **User** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more Users
    * const users = await prisma.user.findMany()
    * ```
    */
  get user(): Prisma.UserDelegate<ExtArgs, ClientOptions>;

  /**
   * `prisma.trainingRun`: Exposes CRUD operations for the **TrainingRun** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more TrainingRuns
    * const trainingRuns = await prisma.trainingRun.findMany()
    * ```
    */
  get trainingRun(): Prisma.TrainingRunDelegate<ExtArgs, ClientOptions>;

  /**
   * `prisma.dataset`: Exposes CRUD operations for the **Dataset** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more Datasets
    * const datasets = await prisma.dataset.findMany()
    * ```
    */
  get dataset(): Prisma.DatasetDelegate<ExtArgs, ClientOptions>;
}

export namespace Prisma {
  export import DMMF = runtime.DMMF

  export type PrismaPromise<T> = $Public.PrismaPromise<T>

  /**
   * Validator
   */
  export import validator = runtime.Public.validator

  /**
   * Prisma Errors
   */
  export import PrismaClientKnownRequestError = runtime.PrismaClientKnownRequestError
  export import PrismaClientUnknownRequestError = runtime.PrismaClientUnknownRequestError
  export import PrismaClientRustPanicError = runtime.PrismaClientRustPanicError
  export import PrismaClientInitializationError = runtime.PrismaClientInitializationError
  export import PrismaClientValidationError = runtime.PrismaClientValidationError

  /**
   * Re-export of sql-template-tag
   */
  export import sql = runtime.sqltag
  export import empty = runtime.empty
  export import join = runtime.join
  export import raw = runtime.raw
  export import Sql = runtime.Sql



  /**
   * Decimal.js
   */
  export import Decimal = runtime.Decimal

  export type DecimalJsLike = runtime.DecimalJsLike

  /**
   * Metrics
   */
  export type Metrics = runtime.Metrics
  export type Metric<T> = runtime.Metric<T>
  export type MetricHistogram = runtime.MetricHistogram
  export type MetricHistogramBucket = runtime.MetricHistogramBucket

  /**
  * Extensions
  */
  export import Extension = $Extensions.UserArgs
  export import getExtensionContext = runtime.Extensions.getExtensionContext
  export import Args = $Public.Args
  export import Payload = $Public.Payload
  export import Result = $Public.Result
  export import Exact = $Public.Exact

  /**
   * Prisma Client JS version: 6.19.0
   * Query Engine version: 393aa359c9ad4a4bb28630fb5613f9c281cde053
   */
  export type PrismaVersion = {
    client: string
  }

  export const prismaVersion: PrismaVersion

  /**
   * Utility Types
   */


  export import Bytes = runtime.Bytes
  export import JsonObject = runtime.JsonObject
  export import JsonArray = runtime.JsonArray
  export import JsonValue = runtime.JsonValue
  export import InputJsonObject = runtime.InputJsonObject
  export import InputJsonArray = runtime.InputJsonArray
  export import InputJsonValue = runtime.InputJsonValue

  /**
   * Types of the values used to represent different kinds of `null` values when working with JSON fields.
   *
   * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
   */
  namespace NullTypes {
    /**
    * Type of `Prisma.DbNull`.
    *
    * You cannot use other instances of this class. Please use the `Prisma.DbNull` value.
    *
    * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
    */
    class DbNull {
      private DbNull: never
      private constructor()
    }

    /**
    * Type of `Prisma.JsonNull`.
    *
    * You cannot use other instances of this class. Please use the `Prisma.JsonNull` value.
    *
    * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
    */
    class JsonNull {
      private JsonNull: never
      private constructor()
    }

    /**
    * Type of `Prisma.AnyNull`.
    *
    * You cannot use other instances of this class. Please use the `Prisma.AnyNull` value.
    *
    * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
    */
    class AnyNull {
      private AnyNull: never
      private constructor()
    }
  }

  /**
   * Helper for filtering JSON entries that have `null` on the database (empty on the db)
   *
   * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
   */
  export const DbNull: NullTypes.DbNull

  /**
   * Helper for filtering JSON entries that have JSON `null` values (not empty on the db)
   *
   * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
   */
  export const JsonNull: NullTypes.JsonNull

  /**
   * Helper for filtering JSON entries that are `Prisma.DbNull` or `Prisma.JsonNull`
   *
   * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
   */
  export const AnyNull: NullTypes.AnyNull

  type SelectAndInclude = {
    select: any
    include: any
  }

  type SelectAndOmit = {
    select: any
    omit: any
  }

  /**
   * Get the type of the value, that the Promise holds.
   */
  export type PromiseType<T extends PromiseLike<any>> = T extends PromiseLike<infer U> ? U : T;

  /**
   * Get the return type of a function which returns a Promise.
   */
  export type PromiseReturnType<T extends (...args: any) => $Utils.JsPromise<any>> = PromiseType<ReturnType<T>>

  /**
   * From T, pick a set of properties whose keys are in the union K
   */
  type Prisma__Pick<T, K extends keyof T> = {
      [P in K]: T[P];
  };


  export type Enumerable<T> = T | Array<T>;

  export type RequiredKeys<T> = {
    [K in keyof T]-?: {} extends Prisma__Pick<T, K> ? never : K
  }[keyof T]

  export type TruthyKeys<T> = keyof {
    [K in keyof T as T[K] extends false | undefined | null ? never : K]: K
  }

  export type TrueKeys<T> = TruthyKeys<Prisma__Pick<T, RequiredKeys<T>>>

  /**
   * Subset
   * @desc From `T` pick properties that exist in `U`. Simple version of Intersection
   */
  export type Subset<T, U> = {
    [key in keyof T]: key extends keyof U ? T[key] : never;
  };

  /**
   * SelectSubset
   * @desc From `T` pick properties that exist in `U`. Simple version of Intersection.
   * Additionally, it validates, if both select and include are present. If the case, it errors.
   */
  export type SelectSubset<T, U> = {
    [key in keyof T]: key extends keyof U ? T[key] : never
  } &
    (T extends SelectAndInclude
      ? 'Please either choose `select` or `include`.'
      : T extends SelectAndOmit
        ? 'Please either choose `select` or `omit`.'
        : {})

  /**
   * Subset + Intersection
   * @desc From `T` pick properties that exist in `U` and intersect `K`
   */
  export type SubsetIntersection<T, U, K> = {
    [key in keyof T]: key extends keyof U ? T[key] : never
  } &
    K

  type Without<T, U> = { [P in Exclude<keyof T, keyof U>]?: never };

  /**
   * XOR is needed to have a real mutually exclusive union type
   * https://stackoverflow.com/questions/42123407/does-typescript-support-mutually-exclusive-types
   */
  type XOR<T, U> =
    T extends object ?
    U extends object ?
      (Without<T, U> & U) | (Without<U, T> & T)
    : U : T


  /**
   * Is T a Record?
   */
  type IsObject<T extends any> = T extends Array<any>
  ? False
  : T extends Date
  ? False
  : T extends Uint8Array
  ? False
  : T extends BigInt
  ? False
  : T extends object
  ? True
  : False


  /**
   * If it's T[], return T
   */
  export type UnEnumerate<T extends unknown> = T extends Array<infer U> ? U : T

  /**
   * From ts-toolbelt
   */

  type __Either<O extends object, K extends Key> = Omit<O, K> &
    {
      // Merge all but K
      [P in K]: Prisma__Pick<O, P & keyof O> // With K possibilities
    }[K]

  type EitherStrict<O extends object, K extends Key> = Strict<__Either<O, K>>

  type EitherLoose<O extends object, K extends Key> = ComputeRaw<__Either<O, K>>

  type _Either<
    O extends object,
    K extends Key,
    strict extends Boolean
  > = {
    1: EitherStrict<O, K>
    0: EitherLoose<O, K>
  }[strict]

  type Either<
    O extends object,
    K extends Key,
    strict extends Boolean = 1
  > = O extends unknown ? _Either<O, K, strict> : never

  export type Union = any

  type PatchUndefined<O extends object, O1 extends object> = {
    [K in keyof O]: O[K] extends undefined ? At<O1, K> : O[K]
  } & {}

  /** Helper Types for "Merge" **/
  export type IntersectOf<U extends Union> = (
    U extends unknown ? (k: U) => void : never
  ) extends (k: infer I) => void
    ? I
    : never

  export type Overwrite<O extends object, O1 extends object> = {
      [K in keyof O]: K extends keyof O1 ? O1[K] : O[K];
  } & {};

  type _Merge<U extends object> = IntersectOf<Overwrite<U, {
      [K in keyof U]-?: At<U, K>;
  }>>;

  type Key = string | number | symbol;
  type AtBasic<O extends object, K extends Key> = K extends keyof O ? O[K] : never;
  type AtStrict<O extends object, K extends Key> = O[K & keyof O];
  type AtLoose<O extends object, K extends Key> = O extends unknown ? AtStrict<O, K> : never;
  export type At<O extends object, K extends Key, strict extends Boolean = 1> = {
      1: AtStrict<O, K>;
      0: AtLoose<O, K>;
  }[strict];

  export type ComputeRaw<A extends any> = A extends Function ? A : {
    [K in keyof A]: A[K];
  } & {};

  export type OptionalFlat<O> = {
    [K in keyof O]?: O[K];
  } & {};

  type _Record<K extends keyof any, T> = {
    [P in K]: T;
  };

  // cause typescript not to expand types and preserve names
  type NoExpand<T> = T extends unknown ? T : never;

  // this type assumes the passed object is entirely optional
  type AtLeast<O extends object, K extends string> = NoExpand<
    O extends unknown
    ? | (K extends keyof O ? { [P in K]: O[P] } & O : O)
      | {[P in keyof O as P extends K ? P : never]-?: O[P]} & O
    : never>;

  type _Strict<U, _U = U> = U extends unknown ? U & OptionalFlat<_Record<Exclude<Keys<_U>, keyof U>, never>> : never;

  export type Strict<U extends object> = ComputeRaw<_Strict<U>>;
  /** End Helper Types for "Merge" **/

  export type Merge<U extends object> = ComputeRaw<_Merge<Strict<U>>>;

  /**
  A [[Boolean]]
  */
  export type Boolean = True | False

  // /**
  // 1
  // */
  export type True = 1

  /**
  0
  */
  export type False = 0

  export type Not<B extends Boolean> = {
    0: 1
    1: 0
  }[B]

  export type Extends<A1 extends any, A2 extends any> = [A1] extends [never]
    ? 0 // anything `never` is false
    : A1 extends A2
    ? 1
    : 0

  export type Has<U extends Union, U1 extends Union> = Not<
    Extends<Exclude<U1, U>, U1>
  >

  export type Or<B1 extends Boolean, B2 extends Boolean> = {
    0: {
      0: 0
      1: 1
    }
    1: {
      0: 1
      1: 1
    }
  }[B1][B2]

  export type Keys<U extends Union> = U extends unknown ? keyof U : never

  type Cast<A, B> = A extends B ? A : B;

  export const type: unique symbol;



  /**
   * Used by group by
   */

  export type GetScalarType<T, O> = O extends object ? {
    [P in keyof T]: P extends keyof O
      ? O[P]
      : never
  } : never

  type FieldPaths<
    T,
    U = Omit<T, '_avg' | '_sum' | '_count' | '_min' | '_max'>
  > = IsObject<T> extends True ? U : T

  type GetHavingFields<T> = {
    [K in keyof T]: Or<
      Or<Extends<'OR', K>, Extends<'AND', K>>,
      Extends<'NOT', K>
    > extends True
      ? // infer is only needed to not hit TS limit
        // based on the brilliant idea of Pierre-Antoine Mills
        // https://github.com/microsoft/TypeScript/issues/30188#issuecomment-478938437
        T[K] extends infer TK
        ? GetHavingFields<UnEnumerate<TK> extends object ? Merge<UnEnumerate<TK>> : never>
        : never
      : {} extends FieldPaths<T[K]>
      ? never
      : K
  }[keyof T]

  /**
   * Convert tuple to union
   */
  type _TupleToUnion<T> = T extends (infer E)[] ? E : never
  type TupleToUnion<K extends readonly any[]> = _TupleToUnion<K>
  type MaybeTupleToUnion<T> = T extends any[] ? TupleToUnion<T> : T

  /**
   * Like `Pick`, but additionally can also accept an array of keys
   */
  type PickEnumerable<T, K extends Enumerable<keyof T> | keyof T> = Prisma__Pick<T, MaybeTupleToUnion<K>>

  /**
   * Exclude all keys with underscores
   */
  type ExcludeUnderscoreKeys<T extends string> = T extends `_${string}` ? never : T


  export type FieldRef<Model, FieldType> = runtime.FieldRef<Model, FieldType>

  type FieldRefInputType<Model, FieldType> = Model extends never ? never : FieldRef<Model, FieldType>


  export const ModelName: {
    User: 'User',
    TrainingRun: 'TrainingRun',
    Dataset: 'Dataset'
  };

  export type ModelName = (typeof ModelName)[keyof typeof ModelName]


  export type Datasources = {
    db?: Datasource
  }

  interface TypeMapCb<ClientOptions = {}> extends $Utils.Fn<{extArgs: $Extensions.InternalArgs }, $Utils.Record<string, any>> {
    returns: Prisma.TypeMap<this['params']['extArgs'], ClientOptions extends { omit: infer OmitOptions } ? OmitOptions : {}>
  }

  export type TypeMap<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> = {
    globalOmitOptions: {
      omit: GlobalOmitOptions
    }
    meta: {
      modelProps: "user" | "trainingRun" | "dataset"
      txIsolationLevel: Prisma.TransactionIsolationLevel
    }
    model: {
      User: {
        payload: Prisma.$UserPayload<ExtArgs>
        fields: Prisma.UserFieldRefs
        operations: {
          findUnique: {
            args: Prisma.UserFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.UserFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          findFirst: {
            args: Prisma.UserFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.UserFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          findMany: {
            args: Prisma.UserFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>[]
          }
          create: {
            args: Prisma.UserCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          createMany: {
            args: Prisma.UserCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.UserCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>[]
          }
          delete: {
            args: Prisma.UserDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          update: {
            args: Prisma.UserUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          deleteMany: {
            args: Prisma.UserDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.UserUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          upsert: {
            args: Prisma.UserUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          aggregate: {
            args: Prisma.UserAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateUser>
          }
          groupBy: {
            args: Prisma.UserGroupByArgs<ExtArgs>
            result: $Utils.Optional<UserGroupByOutputType>[]
          }
          count: {
            args: Prisma.UserCountArgs<ExtArgs>
            result: $Utils.Optional<UserCountAggregateOutputType> | number
          }
        }
      }
      TrainingRun: {
        payload: Prisma.$TrainingRunPayload<ExtArgs>
        fields: Prisma.TrainingRunFieldRefs
        operations: {
          findUnique: {
            args: Prisma.TrainingRunFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TrainingRunPayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.TrainingRunFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TrainingRunPayload>
          }
          findFirst: {
            args: Prisma.TrainingRunFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TrainingRunPayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.TrainingRunFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TrainingRunPayload>
          }
          findMany: {
            args: Prisma.TrainingRunFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TrainingRunPayload>[]
          }
          create: {
            args: Prisma.TrainingRunCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TrainingRunPayload>
          }
          createMany: {
            args: Prisma.TrainingRunCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.TrainingRunCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TrainingRunPayload>[]
          }
          delete: {
            args: Prisma.TrainingRunDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TrainingRunPayload>
          }
          update: {
            args: Prisma.TrainingRunUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TrainingRunPayload>
          }
          deleteMany: {
            args: Prisma.TrainingRunDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.TrainingRunUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          upsert: {
            args: Prisma.TrainingRunUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TrainingRunPayload>
          }
          aggregate: {
            args: Prisma.TrainingRunAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateTrainingRun>
          }
          groupBy: {
            args: Prisma.TrainingRunGroupByArgs<ExtArgs>
            result: $Utils.Optional<TrainingRunGroupByOutputType>[]
          }
          count: {
            args: Prisma.TrainingRunCountArgs<ExtArgs>
            result: $Utils.Optional<TrainingRunCountAggregateOutputType> | number
          }
        }
      }
      Dataset: {
        payload: Prisma.$DatasetPayload<ExtArgs>
        fields: Prisma.DatasetFieldRefs
        operations: {
          findUnique: {
            args: Prisma.DatasetFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DatasetPayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.DatasetFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DatasetPayload>
          }
          findFirst: {
            args: Prisma.DatasetFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DatasetPayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.DatasetFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DatasetPayload>
          }
          findMany: {
            args: Prisma.DatasetFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DatasetPayload>[]
          }
          create: {
            args: Prisma.DatasetCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DatasetPayload>
          }
          createMany: {
            args: Prisma.DatasetCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.DatasetCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DatasetPayload>[]
          }
          delete: {
            args: Prisma.DatasetDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DatasetPayload>
          }
          update: {
            args: Prisma.DatasetUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DatasetPayload>
          }
          deleteMany: {
            args: Prisma.DatasetDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.DatasetUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          upsert: {
            args: Prisma.DatasetUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DatasetPayload>
          }
          aggregate: {
            args: Prisma.DatasetAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateDataset>
          }
          groupBy: {
            args: Prisma.DatasetGroupByArgs<ExtArgs>
            result: $Utils.Optional<DatasetGroupByOutputType>[]
          }
          count: {
            args: Prisma.DatasetCountArgs<ExtArgs>
            result: $Utils.Optional<DatasetCountAggregateOutputType> | number
          }
        }
      }
    }
  } & {
    other: {
      payload: any
      operations: {
        $executeRaw: {
          args: [query: TemplateStringsArray | Prisma.Sql, ...values: any[]],
          result: any
        }
        $executeRawUnsafe: {
          args: [query: string, ...values: any[]],
          result: any
        }
        $queryRaw: {
          args: [query: TemplateStringsArray | Prisma.Sql, ...values: any[]],
          result: any
        }
        $queryRawUnsafe: {
          args: [query: string, ...values: any[]],
          result: any
        }
      }
    }
  }
  export const defineExtension: $Extensions.ExtendsHook<"define", Prisma.TypeMapCb, $Extensions.DefaultArgs>
  export type DefaultPrismaClient = PrismaClient
  export type ErrorFormat = 'pretty' | 'colorless' | 'minimal'
  export interface PrismaClientOptions {
    /**
     * Overwrites the datasource url from your schema.prisma file
     */
    datasources?: Datasources
    /**
     * Overwrites the datasource url from your schema.prisma file
     */
    datasourceUrl?: string
    /**
     * @default "colorless"
     */
    errorFormat?: ErrorFormat
    /**
     * @example
     * ```
     * // Shorthand for `emit: 'stdout'`
     * log: ['query', 'info', 'warn', 'error']
     * 
     * // Emit as events only
     * log: [
     *   { emit: 'event', level: 'query' },
     *   { emit: 'event', level: 'info' },
     *   { emit: 'event', level: 'warn' }
     *   { emit: 'event', level: 'error' }
     * ]
     * 
     * / Emit as events and log to stdout
     * og: [
     *  { emit: 'stdout', level: 'query' },
     *  { emit: 'stdout', level: 'info' },
     *  { emit: 'stdout', level: 'warn' }
     *  { emit: 'stdout', level: 'error' }
     * 
     * ```
     * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/logging#the-log-option).
     */
    log?: (LogLevel | LogDefinition)[]
    /**
     * The default values for transactionOptions
     * maxWait ?= 2000
     * timeout ?= 5000
     */
    transactionOptions?: {
      maxWait?: number
      timeout?: number
      isolationLevel?: Prisma.TransactionIsolationLevel
    }
    /**
     * Instance of a Driver Adapter, e.g., like one provided by `@prisma/adapter-planetscale`
     */
    adapter?: runtime.SqlDriverAdapterFactory | null
    /**
     * Global configuration for omitting model fields by default.
     * 
     * @example
     * ```
     * const prisma = new PrismaClient({
     *   omit: {
     *     user: {
     *       password: true
     *     }
     *   }
     * })
     * ```
     */
    omit?: Prisma.GlobalOmitConfig
  }
  export type GlobalOmitConfig = {
    user?: UserOmit
    trainingRun?: TrainingRunOmit
    dataset?: DatasetOmit
  }

  /* Types for Logging */
  export type LogLevel = 'info' | 'query' | 'warn' | 'error'
  export type LogDefinition = {
    level: LogLevel
    emit: 'stdout' | 'event'
  }

  export type CheckIsLogLevel<T> = T extends LogLevel ? T : never;

  export type GetLogType<T> = CheckIsLogLevel<
    T extends LogDefinition ? T['level'] : T
  >;

  export type GetEvents<T extends any[]> = T extends Array<LogLevel | LogDefinition>
    ? GetLogType<T[number]>
    : never;

  export type QueryEvent = {
    timestamp: Date
    query: string
    params: string
    duration: number
    target: string
  }

  export type LogEvent = {
    timestamp: Date
    message: string
    target: string
  }
  /* End Types for Logging */


  export type PrismaAction =
    | 'findUnique'
    | 'findUniqueOrThrow'
    | 'findMany'
    | 'findFirst'
    | 'findFirstOrThrow'
    | 'create'
    | 'createMany'
    | 'createManyAndReturn'
    | 'update'
    | 'updateMany'
    | 'updateManyAndReturn'
    | 'upsert'
    | 'delete'
    | 'deleteMany'
    | 'executeRaw'
    | 'queryRaw'
    | 'aggregate'
    | 'count'
    | 'runCommandRaw'
    | 'findRaw'
    | 'groupBy'

  // tested in getLogLevel.test.ts
  export function getLogLevel(log: Array<LogLevel | LogDefinition>): LogLevel | undefined;

  /**
   * `PrismaClient` proxy available in interactive transactions.
   */
  export type TransactionClient = Omit<Prisma.DefaultPrismaClient, runtime.ITXClientDenyList>

  export type Datasource = {
    url?: string
  }

  /**
   * Count Types
   */


  /**
   * Count Type UserCountOutputType
   */

  export type UserCountOutputType = {
    runs: number
    datasets: number
  }

  export type UserCountOutputTypeSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    runs?: boolean | UserCountOutputTypeCountRunsArgs
    datasets?: boolean | UserCountOutputTypeCountDatasetsArgs
  }

  // Custom InputTypes
  /**
   * UserCountOutputType without action
   */
  export type UserCountOutputTypeDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the UserCountOutputType
     */
    select?: UserCountOutputTypeSelect<ExtArgs> | null
  }

  /**
   * UserCountOutputType without action
   */
  export type UserCountOutputTypeCountRunsArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: TrainingRunWhereInput
  }

  /**
   * UserCountOutputType without action
   */
  export type UserCountOutputTypeCountDatasetsArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: DatasetWhereInput
  }


  /**
   * Models
   */

  /**
   * Model User
   */

  export type AggregateUser = {
    _count: UserCountAggregateOutputType | null
    _avg: UserAvgAggregateOutputType | null
    _sum: UserSumAggregateOutputType | null
    _min: UserMinAggregateOutputType | null
    _max: UserMaxAggregateOutputType | null
  }

  export type UserAvgAggregateOutputType = {
    id: number | null
    graduationYear: number | null
  }

  export type UserSumAggregateOutputType = {
    id: number | null
    graduationYear: number | null
  }

  export type UserMinAggregateOutputType = {
    id: number | null
    email: string | null
    name: string | null
    degree: string | null
    field: string | null
    university: string | null
    graduationYear: number | null
    password: string | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type UserMaxAggregateOutputType = {
    id: number | null
    email: string | null
    name: string | null
    degree: string | null
    field: string | null
    university: string | null
    graduationYear: number | null
    password: string | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type UserCountAggregateOutputType = {
    id: number
    email: number
    name: number
    degree: number
    field: number
    university: number
    graduationYear: number
    password: number
    createdAt: number
    updatedAt: number
    _all: number
  }


  export type UserAvgAggregateInputType = {
    id?: true
    graduationYear?: true
  }

  export type UserSumAggregateInputType = {
    id?: true
    graduationYear?: true
  }

  export type UserMinAggregateInputType = {
    id?: true
    email?: true
    name?: true
    degree?: true
    field?: true
    university?: true
    graduationYear?: true
    password?: true
    createdAt?: true
    updatedAt?: true
  }

  export type UserMaxAggregateInputType = {
    id?: true
    email?: true
    name?: true
    degree?: true
    field?: true
    university?: true
    graduationYear?: true
    password?: true
    createdAt?: true
    updatedAt?: true
  }

  export type UserCountAggregateInputType = {
    id?: true
    email?: true
    name?: true
    degree?: true
    field?: true
    university?: true
    graduationYear?: true
    password?: true
    createdAt?: true
    updatedAt?: true
    _all?: true
  }

  export type UserAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which User to aggregate.
     */
    where?: UserWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Users to fetch.
     */
    orderBy?: UserOrderByWithRelationInput | UserOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: UserWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Users from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Users.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned Users
    **/
    _count?: true | UserCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to average
    **/
    _avg?: UserAvgAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to sum
    **/
    _sum?: UserSumAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: UserMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: UserMaxAggregateInputType
  }

  export type GetUserAggregateType<T extends UserAggregateArgs> = {
        [P in keyof T & keyof AggregateUser]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateUser[P]>
      : GetScalarType<T[P], AggregateUser[P]>
  }




  export type UserGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: UserWhereInput
    orderBy?: UserOrderByWithAggregationInput | UserOrderByWithAggregationInput[]
    by: UserScalarFieldEnum[] | UserScalarFieldEnum
    having?: UserScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: UserCountAggregateInputType | true
    _avg?: UserAvgAggregateInputType
    _sum?: UserSumAggregateInputType
    _min?: UserMinAggregateInputType
    _max?: UserMaxAggregateInputType
  }

  export type UserGroupByOutputType = {
    id: number
    email: string
    name: string
    degree: string
    field: string
    university: string
    graduationYear: number
    password: string
    createdAt: Date
    updatedAt: Date
    _count: UserCountAggregateOutputType | null
    _avg: UserAvgAggregateOutputType | null
    _sum: UserSumAggregateOutputType | null
    _min: UserMinAggregateOutputType | null
    _max: UserMaxAggregateOutputType | null
  }

  type GetUserGroupByPayload<T extends UserGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<UserGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof UserGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], UserGroupByOutputType[P]>
            : GetScalarType<T[P], UserGroupByOutputType[P]>
        }
      >
    >


  export type UserSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    email?: boolean
    name?: boolean
    degree?: boolean
    field?: boolean
    university?: boolean
    graduationYear?: boolean
    password?: boolean
    createdAt?: boolean
    updatedAt?: boolean
    runs?: boolean | User$runsArgs<ExtArgs>
    datasets?: boolean | User$datasetsArgs<ExtArgs>
    _count?: boolean | UserCountOutputTypeDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["user"]>

  export type UserSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    email?: boolean
    name?: boolean
    degree?: boolean
    field?: boolean
    university?: boolean
    graduationYear?: boolean
    password?: boolean
    createdAt?: boolean
    updatedAt?: boolean
  }, ExtArgs["result"]["user"]>


  export type UserSelectScalar = {
    id?: boolean
    email?: boolean
    name?: boolean
    degree?: boolean
    field?: boolean
    university?: boolean
    graduationYear?: boolean
    password?: boolean
    createdAt?: boolean
    updatedAt?: boolean
  }

  export type UserOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "email" | "name" | "degree" | "field" | "university" | "graduationYear" | "password" | "createdAt" | "updatedAt", ExtArgs["result"]["user"]>
  export type UserInclude<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    runs?: boolean | User$runsArgs<ExtArgs>
    datasets?: boolean | User$datasetsArgs<ExtArgs>
    _count?: boolean | UserCountOutputTypeDefaultArgs<ExtArgs>
  }
  export type UserIncludeCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {}

  export type $UserPayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "User"
    objects: {
      runs: Prisma.$TrainingRunPayload<ExtArgs>[]
      datasets: Prisma.$DatasetPayload<ExtArgs>[]
    }
    scalars: $Extensions.GetPayloadResult<{
      id: number
      email: string
      name: string
      degree: string
      field: string
      university: string
      graduationYear: number
      password: string
      createdAt: Date
      updatedAt: Date
    }, ExtArgs["result"]["user"]>
    composites: {}
  }

  type UserGetPayload<S extends boolean | null | undefined | UserDefaultArgs> = $Result.GetResult<Prisma.$UserPayload, S>

  type UserCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<UserFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: UserCountAggregateInputType | true
    }

  export interface UserDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['User'], meta: { name: 'User' } }
    /**
     * Find zero or one User that matches the filter.
     * @param {UserFindUniqueArgs} args - Arguments to find a User
     * @example
     * // Get one User
     * const user = await prisma.user.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends UserFindUniqueArgs>(args: SelectSubset<T, UserFindUniqueArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one User that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {UserFindUniqueOrThrowArgs} args - Arguments to find a User
     * @example
     * // Get one User
     * const user = await prisma.user.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends UserFindUniqueOrThrowArgs>(args: SelectSubset<T, UserFindUniqueOrThrowArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first User that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserFindFirstArgs} args - Arguments to find a User
     * @example
     * // Get one User
     * const user = await prisma.user.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends UserFindFirstArgs>(args?: SelectSubset<T, UserFindFirstArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first User that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserFindFirstOrThrowArgs} args - Arguments to find a User
     * @example
     * // Get one User
     * const user = await prisma.user.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends UserFindFirstOrThrowArgs>(args?: SelectSubset<T, UserFindFirstOrThrowArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more Users that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all Users
     * const users = await prisma.user.findMany()
     * 
     * // Get first 10 Users
     * const users = await prisma.user.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const userWithIdOnly = await prisma.user.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends UserFindManyArgs>(args?: SelectSubset<T, UserFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a User.
     * @param {UserCreateArgs} args - Arguments to create a User.
     * @example
     * // Create one User
     * const User = await prisma.user.create({
     *   data: {
     *     // ... data to create a User
     *   }
     * })
     * 
     */
    create<T extends UserCreateArgs>(args: SelectSubset<T, UserCreateArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many Users.
     * @param {UserCreateManyArgs} args - Arguments to create many Users.
     * @example
     * // Create many Users
     * const user = await prisma.user.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends UserCreateManyArgs>(args?: SelectSubset<T, UserCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many Users and returns the data saved in the database.
     * @param {UserCreateManyAndReturnArgs} args - Arguments to create many Users.
     * @example
     * // Create many Users
     * const user = await prisma.user.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many Users and only return the `id`
     * const userWithIdOnly = await prisma.user.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends UserCreateManyAndReturnArgs>(args?: SelectSubset<T, UserCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a User.
     * @param {UserDeleteArgs} args - Arguments to delete one User.
     * @example
     * // Delete one User
     * const User = await prisma.user.delete({
     *   where: {
     *     // ... filter to delete one User
     *   }
     * })
     * 
     */
    delete<T extends UserDeleteArgs>(args: SelectSubset<T, UserDeleteArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one User.
     * @param {UserUpdateArgs} args - Arguments to update one User.
     * @example
     * // Update one User
     * const user = await prisma.user.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends UserUpdateArgs>(args: SelectSubset<T, UserUpdateArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more Users.
     * @param {UserDeleteManyArgs} args - Arguments to filter Users to delete.
     * @example
     * // Delete a few Users
     * const { count } = await prisma.user.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends UserDeleteManyArgs>(args?: SelectSubset<T, UserDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more Users.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many Users
     * const user = await prisma.user.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends UserUpdateManyArgs>(args: SelectSubset<T, UserUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create or update one User.
     * @param {UserUpsertArgs} args - Arguments to update or create a User.
     * @example
     * // Update or create a User
     * const user = await prisma.user.upsert({
     *   create: {
     *     // ... data to create a User
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the User we want to update
     *   }
     * })
     */
    upsert<T extends UserUpsertArgs>(args: SelectSubset<T, UserUpsertArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of Users.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserCountArgs} args - Arguments to filter Users to count.
     * @example
     * // Count the number of Users
     * const count = await prisma.user.count({
     *   where: {
     *     // ... the filter for the Users we want to count
     *   }
     * })
    **/
    count<T extends UserCountArgs>(
      args?: Subset<T, UserCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], UserCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a User.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends UserAggregateArgs>(args: Subset<T, UserAggregateArgs>): Prisma.PrismaPromise<GetUserAggregateType<T>>

    /**
     * Group by User.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends UserGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: UserGroupByArgs['orderBy'] }
        : { orderBy?: UserGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, UserGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetUserGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the User model
   */
  readonly fields: UserFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for User.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__UserClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    runs<T extends User$runsArgs<ExtArgs> = {}>(args?: Subset<T, User$runsArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$TrainingRunPayload<ExtArgs>, T, "findMany", GlobalOmitOptions> | Null>
    datasets<T extends User$datasetsArgs<ExtArgs> = {}>(args?: Subset<T, User$datasetsArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$DatasetPayload<ExtArgs>, T, "findMany", GlobalOmitOptions> | Null>
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the User model
   */
  interface UserFieldRefs {
    readonly id: FieldRef<"User", 'Int'>
    readonly email: FieldRef<"User", 'String'>
    readonly name: FieldRef<"User", 'String'>
    readonly degree: FieldRef<"User", 'String'>
    readonly field: FieldRef<"User", 'String'>
    readonly university: FieldRef<"User", 'String'>
    readonly graduationYear: FieldRef<"User", 'Int'>
    readonly password: FieldRef<"User", 'String'>
    readonly createdAt: FieldRef<"User", 'DateTime'>
    readonly updatedAt: FieldRef<"User", 'DateTime'>
  }
    

  // Custom InputTypes
  /**
   * User findUnique
   */
  export type UserFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * Filter, which User to fetch.
     */
    where: UserWhereUniqueInput
  }

  /**
   * User findUniqueOrThrow
   */
  export type UserFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * Filter, which User to fetch.
     */
    where: UserWhereUniqueInput
  }

  /**
   * User findFirst
   */
  export type UserFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * Filter, which User to fetch.
     */
    where?: UserWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Users to fetch.
     */
    orderBy?: UserOrderByWithRelationInput | UserOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for Users.
     */
    cursor?: UserWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Users from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Users.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of Users.
     */
    distinct?: UserScalarFieldEnum | UserScalarFieldEnum[]
  }

  /**
   * User findFirstOrThrow
   */
  export type UserFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * Filter, which User to fetch.
     */
    where?: UserWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Users to fetch.
     */
    orderBy?: UserOrderByWithRelationInput | UserOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for Users.
     */
    cursor?: UserWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Users from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Users.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of Users.
     */
    distinct?: UserScalarFieldEnum | UserScalarFieldEnum[]
  }

  /**
   * User findMany
   */
  export type UserFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * Filter, which Users to fetch.
     */
    where?: UserWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Users to fetch.
     */
    orderBy?: UserOrderByWithRelationInput | UserOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing Users.
     */
    cursor?: UserWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Users from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Users.
     */
    skip?: number
    distinct?: UserScalarFieldEnum | UserScalarFieldEnum[]
  }

  /**
   * User create
   */
  export type UserCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * The data needed to create a User.
     */
    data: XOR<UserCreateInput, UserUncheckedCreateInput>
  }

  /**
   * User createMany
   */
  export type UserCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many Users.
     */
    data: UserCreateManyInput | UserCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * User createManyAndReturn
   */
  export type UserCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * The data used to create many Users.
     */
    data: UserCreateManyInput | UserCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * User update
   */
  export type UserUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * The data needed to update a User.
     */
    data: XOR<UserUpdateInput, UserUncheckedUpdateInput>
    /**
     * Choose, which User to update.
     */
    where: UserWhereUniqueInput
  }

  /**
   * User updateMany
   */
  export type UserUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update Users.
     */
    data: XOR<UserUpdateManyMutationInput, UserUncheckedUpdateManyInput>
    /**
     * Filter which Users to update
     */
    where?: UserWhereInput
  }

  /**
   * User upsert
   */
  export type UserUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * The filter to search for the User to update in case it exists.
     */
    where: UserWhereUniqueInput
    /**
     * In case the User found by the `where` argument doesn't exist, create a new User with this data.
     */
    create: XOR<UserCreateInput, UserUncheckedCreateInput>
    /**
     * In case the User was found with the provided `where` argument, update it with this data.
     */
    update: XOR<UserUpdateInput, UserUncheckedUpdateInput>
  }

  /**
   * User delete
   */
  export type UserDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * Filter which User to delete.
     */
    where: UserWhereUniqueInput
  }

  /**
   * User deleteMany
   */
  export type UserDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which Users to delete
     */
    where?: UserWhereInput
  }

  /**
   * User.runs
   */
  export type User$runsArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TrainingRun
     */
    select?: TrainingRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TrainingRun
     */
    omit?: TrainingRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TrainingRunInclude<ExtArgs> | null
    where?: TrainingRunWhereInput
    orderBy?: TrainingRunOrderByWithRelationInput | TrainingRunOrderByWithRelationInput[]
    cursor?: TrainingRunWhereUniqueInput
    take?: number
    skip?: number
    distinct?: TrainingRunScalarFieldEnum | TrainingRunScalarFieldEnum[]
  }

  /**
   * User.datasets
   */
  export type User$datasetsArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Dataset
     */
    select?: DatasetSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Dataset
     */
    omit?: DatasetOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DatasetInclude<ExtArgs> | null
    where?: DatasetWhereInput
    orderBy?: DatasetOrderByWithRelationInput | DatasetOrderByWithRelationInput[]
    cursor?: DatasetWhereUniqueInput
    take?: number
    skip?: number
    distinct?: DatasetScalarFieldEnum | DatasetScalarFieldEnum[]
  }

  /**
   * User without action
   */
  export type UserDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
  }


  /**
   * Model TrainingRun
   */

  export type AggregateTrainingRun = {
    _count: TrainingRunCountAggregateOutputType | null
    _avg: TrainingRunAvgAggregateOutputType | null
    _sum: TrainingRunSumAggregateOutputType | null
    _min: TrainingRunMinAggregateOutputType | null
    _max: TrainingRunMaxAggregateOutputType | null
  }

  export type TrainingRunAvgAggregateOutputType = {
    userId: number | null
  }

  export type TrainingRunSumAggregateOutputType = {
    userId: number | null
  }

  export type TrainingRunMinAggregateOutputType = {
    id: string | null
    userId: number | null
    name: string | null
    description: string | null
    status: string | null
    datasetUri: string | null
    modelPath: string | null
    resultsPath: string | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type TrainingRunMaxAggregateOutputType = {
    id: string | null
    userId: number | null
    name: string | null
    description: string | null
    status: string | null
    datasetUri: string | null
    modelPath: string | null
    resultsPath: string | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type TrainingRunCountAggregateOutputType = {
    id: number
    userId: number
    name: number
    description: number
    status: number
    datasetUri: number
    modelPath: number
    resultsPath: number
    metrics: number
    parameters: number
    geneDepmapResults: number
    createdAt: number
    updatedAt: number
    _all: number
  }


  export type TrainingRunAvgAggregateInputType = {
    userId?: true
  }

  export type TrainingRunSumAggregateInputType = {
    userId?: true
  }

  export type TrainingRunMinAggregateInputType = {
    id?: true
    userId?: true
    name?: true
    description?: true
    status?: true
    datasetUri?: true
    modelPath?: true
    resultsPath?: true
    createdAt?: true
    updatedAt?: true
  }

  export type TrainingRunMaxAggregateInputType = {
    id?: true
    userId?: true
    name?: true
    description?: true
    status?: true
    datasetUri?: true
    modelPath?: true
    resultsPath?: true
    createdAt?: true
    updatedAt?: true
  }

  export type TrainingRunCountAggregateInputType = {
    id?: true
    userId?: true
    name?: true
    description?: true
    status?: true
    datasetUri?: true
    modelPath?: true
    resultsPath?: true
    metrics?: true
    parameters?: true
    geneDepmapResults?: true
    createdAt?: true
    updatedAt?: true
    _all?: true
  }

  export type TrainingRunAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which TrainingRun to aggregate.
     */
    where?: TrainingRunWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of TrainingRuns to fetch.
     */
    orderBy?: TrainingRunOrderByWithRelationInput | TrainingRunOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: TrainingRunWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` TrainingRuns from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` TrainingRuns.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned TrainingRuns
    **/
    _count?: true | TrainingRunCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to average
    **/
    _avg?: TrainingRunAvgAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to sum
    **/
    _sum?: TrainingRunSumAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: TrainingRunMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: TrainingRunMaxAggregateInputType
  }

  export type GetTrainingRunAggregateType<T extends TrainingRunAggregateArgs> = {
        [P in keyof T & keyof AggregateTrainingRun]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateTrainingRun[P]>
      : GetScalarType<T[P], AggregateTrainingRun[P]>
  }




  export type TrainingRunGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: TrainingRunWhereInput
    orderBy?: TrainingRunOrderByWithAggregationInput | TrainingRunOrderByWithAggregationInput[]
    by: TrainingRunScalarFieldEnum[] | TrainingRunScalarFieldEnum
    having?: TrainingRunScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: TrainingRunCountAggregateInputType | true
    _avg?: TrainingRunAvgAggregateInputType
    _sum?: TrainingRunSumAggregateInputType
    _min?: TrainingRunMinAggregateInputType
    _max?: TrainingRunMaxAggregateInputType
  }

  export type TrainingRunGroupByOutputType = {
    id: string
    userId: number
    name: string
    description: string
    status: string
    datasetUri: string
    modelPath: string | null
    resultsPath: string | null
    metrics: JsonValue | null
    parameters: JsonValue | null
    geneDepmapResults: JsonValue | null
    createdAt: Date
    updatedAt: Date
    _count: TrainingRunCountAggregateOutputType | null
    _avg: TrainingRunAvgAggregateOutputType | null
    _sum: TrainingRunSumAggregateOutputType | null
    _min: TrainingRunMinAggregateOutputType | null
    _max: TrainingRunMaxAggregateOutputType | null
  }

  type GetTrainingRunGroupByPayload<T extends TrainingRunGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<TrainingRunGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof TrainingRunGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], TrainingRunGroupByOutputType[P]>
            : GetScalarType<T[P], TrainingRunGroupByOutputType[P]>
        }
      >
    >


  export type TrainingRunSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    userId?: boolean
    name?: boolean
    description?: boolean
    status?: boolean
    datasetUri?: boolean
    modelPath?: boolean
    resultsPath?: boolean
    metrics?: boolean
    parameters?: boolean
    geneDepmapResults?: boolean
    createdAt?: boolean
    updatedAt?: boolean
    user?: boolean | UserDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["trainingRun"]>

  export type TrainingRunSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    userId?: boolean
    name?: boolean
    description?: boolean
    status?: boolean
    datasetUri?: boolean
    modelPath?: boolean
    resultsPath?: boolean
    metrics?: boolean
    parameters?: boolean
    geneDepmapResults?: boolean
    createdAt?: boolean
    updatedAt?: boolean
    user?: boolean | UserDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["trainingRun"]>


  export type TrainingRunSelectScalar = {
    id?: boolean
    userId?: boolean
    name?: boolean
    description?: boolean
    status?: boolean
    datasetUri?: boolean
    modelPath?: boolean
    resultsPath?: boolean
    metrics?: boolean
    parameters?: boolean
    geneDepmapResults?: boolean
    createdAt?: boolean
    updatedAt?: boolean
  }

  export type TrainingRunOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "userId" | "name" | "description" | "status" | "datasetUri" | "modelPath" | "resultsPath" | "metrics" | "parameters" | "geneDepmapResults" | "createdAt" | "updatedAt", ExtArgs["result"]["trainingRun"]>
  export type TrainingRunInclude<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    user?: boolean | UserDefaultArgs<ExtArgs>
  }
  export type TrainingRunIncludeCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    user?: boolean | UserDefaultArgs<ExtArgs>
  }

  export type $TrainingRunPayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "TrainingRun"
    objects: {
      user: Prisma.$UserPayload<ExtArgs>
    }
    scalars: $Extensions.GetPayloadResult<{
      id: string
      userId: number
      name: string
      description: string
      status: string
      datasetUri: string
      modelPath: string | null
      resultsPath: string | null
      metrics: Prisma.JsonValue | null
      parameters: Prisma.JsonValue | null
      geneDepmapResults: Prisma.JsonValue | null
      createdAt: Date
      updatedAt: Date
    }, ExtArgs["result"]["trainingRun"]>
    composites: {}
  }

  type TrainingRunGetPayload<S extends boolean | null | undefined | TrainingRunDefaultArgs> = $Result.GetResult<Prisma.$TrainingRunPayload, S>

  type TrainingRunCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<TrainingRunFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: TrainingRunCountAggregateInputType | true
    }

  export interface TrainingRunDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['TrainingRun'], meta: { name: 'TrainingRun' } }
    /**
     * Find zero or one TrainingRun that matches the filter.
     * @param {TrainingRunFindUniqueArgs} args - Arguments to find a TrainingRun
     * @example
     * // Get one TrainingRun
     * const trainingRun = await prisma.trainingRun.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends TrainingRunFindUniqueArgs>(args: SelectSubset<T, TrainingRunFindUniqueArgs<ExtArgs>>): Prisma__TrainingRunClient<$Result.GetResult<Prisma.$TrainingRunPayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one TrainingRun that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {TrainingRunFindUniqueOrThrowArgs} args - Arguments to find a TrainingRun
     * @example
     * // Get one TrainingRun
     * const trainingRun = await prisma.trainingRun.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends TrainingRunFindUniqueOrThrowArgs>(args: SelectSubset<T, TrainingRunFindUniqueOrThrowArgs<ExtArgs>>): Prisma__TrainingRunClient<$Result.GetResult<Prisma.$TrainingRunPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first TrainingRun that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TrainingRunFindFirstArgs} args - Arguments to find a TrainingRun
     * @example
     * // Get one TrainingRun
     * const trainingRun = await prisma.trainingRun.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends TrainingRunFindFirstArgs>(args?: SelectSubset<T, TrainingRunFindFirstArgs<ExtArgs>>): Prisma__TrainingRunClient<$Result.GetResult<Prisma.$TrainingRunPayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first TrainingRun that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TrainingRunFindFirstOrThrowArgs} args - Arguments to find a TrainingRun
     * @example
     * // Get one TrainingRun
     * const trainingRun = await prisma.trainingRun.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends TrainingRunFindFirstOrThrowArgs>(args?: SelectSubset<T, TrainingRunFindFirstOrThrowArgs<ExtArgs>>): Prisma__TrainingRunClient<$Result.GetResult<Prisma.$TrainingRunPayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more TrainingRuns that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TrainingRunFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all TrainingRuns
     * const trainingRuns = await prisma.trainingRun.findMany()
     * 
     * // Get first 10 TrainingRuns
     * const trainingRuns = await prisma.trainingRun.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const trainingRunWithIdOnly = await prisma.trainingRun.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends TrainingRunFindManyArgs>(args?: SelectSubset<T, TrainingRunFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$TrainingRunPayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a TrainingRun.
     * @param {TrainingRunCreateArgs} args - Arguments to create a TrainingRun.
     * @example
     * // Create one TrainingRun
     * const TrainingRun = await prisma.trainingRun.create({
     *   data: {
     *     // ... data to create a TrainingRun
     *   }
     * })
     * 
     */
    create<T extends TrainingRunCreateArgs>(args: SelectSubset<T, TrainingRunCreateArgs<ExtArgs>>): Prisma__TrainingRunClient<$Result.GetResult<Prisma.$TrainingRunPayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many TrainingRuns.
     * @param {TrainingRunCreateManyArgs} args - Arguments to create many TrainingRuns.
     * @example
     * // Create many TrainingRuns
     * const trainingRun = await prisma.trainingRun.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends TrainingRunCreateManyArgs>(args?: SelectSubset<T, TrainingRunCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many TrainingRuns and returns the data saved in the database.
     * @param {TrainingRunCreateManyAndReturnArgs} args - Arguments to create many TrainingRuns.
     * @example
     * // Create many TrainingRuns
     * const trainingRun = await prisma.trainingRun.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many TrainingRuns and only return the `id`
     * const trainingRunWithIdOnly = await prisma.trainingRun.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends TrainingRunCreateManyAndReturnArgs>(args?: SelectSubset<T, TrainingRunCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$TrainingRunPayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a TrainingRun.
     * @param {TrainingRunDeleteArgs} args - Arguments to delete one TrainingRun.
     * @example
     * // Delete one TrainingRun
     * const TrainingRun = await prisma.trainingRun.delete({
     *   where: {
     *     // ... filter to delete one TrainingRun
     *   }
     * })
     * 
     */
    delete<T extends TrainingRunDeleteArgs>(args: SelectSubset<T, TrainingRunDeleteArgs<ExtArgs>>): Prisma__TrainingRunClient<$Result.GetResult<Prisma.$TrainingRunPayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one TrainingRun.
     * @param {TrainingRunUpdateArgs} args - Arguments to update one TrainingRun.
     * @example
     * // Update one TrainingRun
     * const trainingRun = await prisma.trainingRun.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends TrainingRunUpdateArgs>(args: SelectSubset<T, TrainingRunUpdateArgs<ExtArgs>>): Prisma__TrainingRunClient<$Result.GetResult<Prisma.$TrainingRunPayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more TrainingRuns.
     * @param {TrainingRunDeleteManyArgs} args - Arguments to filter TrainingRuns to delete.
     * @example
     * // Delete a few TrainingRuns
     * const { count } = await prisma.trainingRun.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends TrainingRunDeleteManyArgs>(args?: SelectSubset<T, TrainingRunDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more TrainingRuns.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TrainingRunUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many TrainingRuns
     * const trainingRun = await prisma.trainingRun.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends TrainingRunUpdateManyArgs>(args: SelectSubset<T, TrainingRunUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create or update one TrainingRun.
     * @param {TrainingRunUpsertArgs} args - Arguments to update or create a TrainingRun.
     * @example
     * // Update or create a TrainingRun
     * const trainingRun = await prisma.trainingRun.upsert({
     *   create: {
     *     // ... data to create a TrainingRun
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the TrainingRun we want to update
     *   }
     * })
     */
    upsert<T extends TrainingRunUpsertArgs>(args: SelectSubset<T, TrainingRunUpsertArgs<ExtArgs>>): Prisma__TrainingRunClient<$Result.GetResult<Prisma.$TrainingRunPayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of TrainingRuns.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TrainingRunCountArgs} args - Arguments to filter TrainingRuns to count.
     * @example
     * // Count the number of TrainingRuns
     * const count = await prisma.trainingRun.count({
     *   where: {
     *     // ... the filter for the TrainingRuns we want to count
     *   }
     * })
    **/
    count<T extends TrainingRunCountArgs>(
      args?: Subset<T, TrainingRunCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], TrainingRunCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a TrainingRun.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TrainingRunAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends TrainingRunAggregateArgs>(args: Subset<T, TrainingRunAggregateArgs>): Prisma.PrismaPromise<GetTrainingRunAggregateType<T>>

    /**
     * Group by TrainingRun.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TrainingRunGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends TrainingRunGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: TrainingRunGroupByArgs['orderBy'] }
        : { orderBy?: TrainingRunGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, TrainingRunGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetTrainingRunGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the TrainingRun model
   */
  readonly fields: TrainingRunFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for TrainingRun.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__TrainingRunClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    user<T extends UserDefaultArgs<ExtArgs> = {}>(args?: Subset<T, UserDefaultArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions> | Null, Null, ExtArgs, GlobalOmitOptions>
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the TrainingRun model
   */
  interface TrainingRunFieldRefs {
    readonly id: FieldRef<"TrainingRun", 'String'>
    readonly userId: FieldRef<"TrainingRun", 'Int'>
    readonly name: FieldRef<"TrainingRun", 'String'>
    readonly description: FieldRef<"TrainingRun", 'String'>
    readonly status: FieldRef<"TrainingRun", 'String'>
    readonly datasetUri: FieldRef<"TrainingRun", 'String'>
    readonly modelPath: FieldRef<"TrainingRun", 'String'>
    readonly resultsPath: FieldRef<"TrainingRun", 'String'>
    readonly metrics: FieldRef<"TrainingRun", 'Json'>
    readonly parameters: FieldRef<"TrainingRun", 'Json'>
    readonly geneDepmapResults: FieldRef<"TrainingRun", 'Json'>
    readonly createdAt: FieldRef<"TrainingRun", 'DateTime'>
    readonly updatedAt: FieldRef<"TrainingRun", 'DateTime'>
  }
    

  // Custom InputTypes
  /**
   * TrainingRun findUnique
   */
  export type TrainingRunFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TrainingRun
     */
    select?: TrainingRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TrainingRun
     */
    omit?: TrainingRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TrainingRunInclude<ExtArgs> | null
    /**
     * Filter, which TrainingRun to fetch.
     */
    where: TrainingRunWhereUniqueInput
  }

  /**
   * TrainingRun findUniqueOrThrow
   */
  export type TrainingRunFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TrainingRun
     */
    select?: TrainingRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TrainingRun
     */
    omit?: TrainingRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TrainingRunInclude<ExtArgs> | null
    /**
     * Filter, which TrainingRun to fetch.
     */
    where: TrainingRunWhereUniqueInput
  }

  /**
   * TrainingRun findFirst
   */
  export type TrainingRunFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TrainingRun
     */
    select?: TrainingRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TrainingRun
     */
    omit?: TrainingRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TrainingRunInclude<ExtArgs> | null
    /**
     * Filter, which TrainingRun to fetch.
     */
    where?: TrainingRunWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of TrainingRuns to fetch.
     */
    orderBy?: TrainingRunOrderByWithRelationInput | TrainingRunOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for TrainingRuns.
     */
    cursor?: TrainingRunWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` TrainingRuns from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` TrainingRuns.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of TrainingRuns.
     */
    distinct?: TrainingRunScalarFieldEnum | TrainingRunScalarFieldEnum[]
  }

  /**
   * TrainingRun findFirstOrThrow
   */
  export type TrainingRunFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TrainingRun
     */
    select?: TrainingRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TrainingRun
     */
    omit?: TrainingRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TrainingRunInclude<ExtArgs> | null
    /**
     * Filter, which TrainingRun to fetch.
     */
    where?: TrainingRunWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of TrainingRuns to fetch.
     */
    orderBy?: TrainingRunOrderByWithRelationInput | TrainingRunOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for TrainingRuns.
     */
    cursor?: TrainingRunWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` TrainingRuns from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` TrainingRuns.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of TrainingRuns.
     */
    distinct?: TrainingRunScalarFieldEnum | TrainingRunScalarFieldEnum[]
  }

  /**
   * TrainingRun findMany
   */
  export type TrainingRunFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TrainingRun
     */
    select?: TrainingRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TrainingRun
     */
    omit?: TrainingRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TrainingRunInclude<ExtArgs> | null
    /**
     * Filter, which TrainingRuns to fetch.
     */
    where?: TrainingRunWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of TrainingRuns to fetch.
     */
    orderBy?: TrainingRunOrderByWithRelationInput | TrainingRunOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing TrainingRuns.
     */
    cursor?: TrainingRunWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` TrainingRuns from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` TrainingRuns.
     */
    skip?: number
    distinct?: TrainingRunScalarFieldEnum | TrainingRunScalarFieldEnum[]
  }

  /**
   * TrainingRun create
   */
  export type TrainingRunCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TrainingRun
     */
    select?: TrainingRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TrainingRun
     */
    omit?: TrainingRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TrainingRunInclude<ExtArgs> | null
    /**
     * The data needed to create a TrainingRun.
     */
    data: XOR<TrainingRunCreateInput, TrainingRunUncheckedCreateInput>
  }

  /**
   * TrainingRun createMany
   */
  export type TrainingRunCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many TrainingRuns.
     */
    data: TrainingRunCreateManyInput | TrainingRunCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * TrainingRun createManyAndReturn
   */
  export type TrainingRunCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TrainingRun
     */
    select?: TrainingRunSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the TrainingRun
     */
    omit?: TrainingRunOmit<ExtArgs> | null
    /**
     * The data used to create many TrainingRuns.
     */
    data: TrainingRunCreateManyInput | TrainingRunCreateManyInput[]
    skipDuplicates?: boolean
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TrainingRunIncludeCreateManyAndReturn<ExtArgs> | null
  }

  /**
   * TrainingRun update
   */
  export type TrainingRunUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TrainingRun
     */
    select?: TrainingRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TrainingRun
     */
    omit?: TrainingRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TrainingRunInclude<ExtArgs> | null
    /**
     * The data needed to update a TrainingRun.
     */
    data: XOR<TrainingRunUpdateInput, TrainingRunUncheckedUpdateInput>
    /**
     * Choose, which TrainingRun to update.
     */
    where: TrainingRunWhereUniqueInput
  }

  /**
   * TrainingRun updateMany
   */
  export type TrainingRunUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update TrainingRuns.
     */
    data: XOR<TrainingRunUpdateManyMutationInput, TrainingRunUncheckedUpdateManyInput>
    /**
     * Filter which TrainingRuns to update
     */
    where?: TrainingRunWhereInput
  }

  /**
   * TrainingRun upsert
   */
  export type TrainingRunUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TrainingRun
     */
    select?: TrainingRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TrainingRun
     */
    omit?: TrainingRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TrainingRunInclude<ExtArgs> | null
    /**
     * The filter to search for the TrainingRun to update in case it exists.
     */
    where: TrainingRunWhereUniqueInput
    /**
     * In case the TrainingRun found by the `where` argument doesn't exist, create a new TrainingRun with this data.
     */
    create: XOR<TrainingRunCreateInput, TrainingRunUncheckedCreateInput>
    /**
     * In case the TrainingRun was found with the provided `where` argument, update it with this data.
     */
    update: XOR<TrainingRunUpdateInput, TrainingRunUncheckedUpdateInput>
  }

  /**
   * TrainingRun delete
   */
  export type TrainingRunDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TrainingRun
     */
    select?: TrainingRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TrainingRun
     */
    omit?: TrainingRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TrainingRunInclude<ExtArgs> | null
    /**
     * Filter which TrainingRun to delete.
     */
    where: TrainingRunWhereUniqueInput
  }

  /**
   * TrainingRun deleteMany
   */
  export type TrainingRunDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which TrainingRuns to delete
     */
    where?: TrainingRunWhereInput
  }

  /**
   * TrainingRun without action
   */
  export type TrainingRunDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TrainingRun
     */
    select?: TrainingRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TrainingRun
     */
    omit?: TrainingRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TrainingRunInclude<ExtArgs> | null
  }


  /**
   * Model Dataset
   */

  export type AggregateDataset = {
    _count: DatasetCountAggregateOutputType | null
    _avg: DatasetAvgAggregateOutputType | null
    _sum: DatasetSumAggregateOutputType | null
    _min: DatasetMinAggregateOutputType | null
    _max: DatasetMaxAggregateOutputType | null
  }

  export type DatasetAvgAggregateOutputType = {
    userId: number | null
    rowCount: number | null
    columnCount: number | null
  }

  export type DatasetSumAggregateOutputType = {
    userId: number | null
    rowCount: number | null
    columnCount: number | null
  }

  export type DatasetMinAggregateOutputType = {
    id: string | null
    userId: number | null
    filePath: string | null
    name: string | null
    description: string | null
    rowCount: number | null
    columnCount: number | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type DatasetMaxAggregateOutputType = {
    id: string | null
    userId: number | null
    filePath: string | null
    name: string | null
    description: string | null
    rowCount: number | null
    columnCount: number | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type DatasetCountAggregateOutputType = {
    id: number
    userId: number
    filePath: number
    name: number
    description: number
    rowCount: number
    columnCount: number
    createdAt: number
    updatedAt: number
    _all: number
  }


  export type DatasetAvgAggregateInputType = {
    userId?: true
    rowCount?: true
    columnCount?: true
  }

  export type DatasetSumAggregateInputType = {
    userId?: true
    rowCount?: true
    columnCount?: true
  }

  export type DatasetMinAggregateInputType = {
    id?: true
    userId?: true
    filePath?: true
    name?: true
    description?: true
    rowCount?: true
    columnCount?: true
    createdAt?: true
    updatedAt?: true
  }

  export type DatasetMaxAggregateInputType = {
    id?: true
    userId?: true
    filePath?: true
    name?: true
    description?: true
    rowCount?: true
    columnCount?: true
    createdAt?: true
    updatedAt?: true
  }

  export type DatasetCountAggregateInputType = {
    id?: true
    userId?: true
    filePath?: true
    name?: true
    description?: true
    rowCount?: true
    columnCount?: true
    createdAt?: true
    updatedAt?: true
    _all?: true
  }

  export type DatasetAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which Dataset to aggregate.
     */
    where?: DatasetWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Datasets to fetch.
     */
    orderBy?: DatasetOrderByWithRelationInput | DatasetOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: DatasetWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Datasets from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Datasets.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned Datasets
    **/
    _count?: true | DatasetCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to average
    **/
    _avg?: DatasetAvgAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to sum
    **/
    _sum?: DatasetSumAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: DatasetMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: DatasetMaxAggregateInputType
  }

  export type GetDatasetAggregateType<T extends DatasetAggregateArgs> = {
        [P in keyof T & keyof AggregateDataset]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateDataset[P]>
      : GetScalarType<T[P], AggregateDataset[P]>
  }




  export type DatasetGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: DatasetWhereInput
    orderBy?: DatasetOrderByWithAggregationInput | DatasetOrderByWithAggregationInput[]
    by: DatasetScalarFieldEnum[] | DatasetScalarFieldEnum
    having?: DatasetScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: DatasetCountAggregateInputType | true
    _avg?: DatasetAvgAggregateInputType
    _sum?: DatasetSumAggregateInputType
    _min?: DatasetMinAggregateInputType
    _max?: DatasetMaxAggregateInputType
  }

  export type DatasetGroupByOutputType = {
    id: string
    userId: number
    filePath: string
    name: string
    description: string
    rowCount: number | null
    columnCount: number | null
    createdAt: Date
    updatedAt: Date
    _count: DatasetCountAggregateOutputType | null
    _avg: DatasetAvgAggregateOutputType | null
    _sum: DatasetSumAggregateOutputType | null
    _min: DatasetMinAggregateOutputType | null
    _max: DatasetMaxAggregateOutputType | null
  }

  type GetDatasetGroupByPayload<T extends DatasetGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<DatasetGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof DatasetGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], DatasetGroupByOutputType[P]>
            : GetScalarType<T[P], DatasetGroupByOutputType[P]>
        }
      >
    >


  export type DatasetSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    userId?: boolean
    filePath?: boolean
    name?: boolean
    description?: boolean
    rowCount?: boolean
    columnCount?: boolean
    createdAt?: boolean
    updatedAt?: boolean
    user?: boolean | UserDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["dataset"]>

  export type DatasetSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    userId?: boolean
    filePath?: boolean
    name?: boolean
    description?: boolean
    rowCount?: boolean
    columnCount?: boolean
    createdAt?: boolean
    updatedAt?: boolean
    user?: boolean | UserDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["dataset"]>


  export type DatasetSelectScalar = {
    id?: boolean
    userId?: boolean
    filePath?: boolean
    name?: boolean
    description?: boolean
    rowCount?: boolean
    columnCount?: boolean
    createdAt?: boolean
    updatedAt?: boolean
  }

  export type DatasetOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "userId" | "filePath" | "name" | "description" | "rowCount" | "columnCount" | "createdAt" | "updatedAt", ExtArgs["result"]["dataset"]>
  export type DatasetInclude<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    user?: boolean | UserDefaultArgs<ExtArgs>
  }
  export type DatasetIncludeCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    user?: boolean | UserDefaultArgs<ExtArgs>
  }

  export type $DatasetPayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "Dataset"
    objects: {
      user: Prisma.$UserPayload<ExtArgs>
    }
    scalars: $Extensions.GetPayloadResult<{
      id: string
      userId: number
      filePath: string
      name: string
      description: string
      rowCount: number | null
      columnCount: number | null
      createdAt: Date
      updatedAt: Date
    }, ExtArgs["result"]["dataset"]>
    composites: {}
  }

  type DatasetGetPayload<S extends boolean | null | undefined | DatasetDefaultArgs> = $Result.GetResult<Prisma.$DatasetPayload, S>

  type DatasetCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<DatasetFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: DatasetCountAggregateInputType | true
    }

  export interface DatasetDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['Dataset'], meta: { name: 'Dataset' } }
    /**
     * Find zero or one Dataset that matches the filter.
     * @param {DatasetFindUniqueArgs} args - Arguments to find a Dataset
     * @example
     * // Get one Dataset
     * const dataset = await prisma.dataset.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends DatasetFindUniqueArgs>(args: SelectSubset<T, DatasetFindUniqueArgs<ExtArgs>>): Prisma__DatasetClient<$Result.GetResult<Prisma.$DatasetPayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one Dataset that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {DatasetFindUniqueOrThrowArgs} args - Arguments to find a Dataset
     * @example
     * // Get one Dataset
     * const dataset = await prisma.dataset.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends DatasetFindUniqueOrThrowArgs>(args: SelectSubset<T, DatasetFindUniqueOrThrowArgs<ExtArgs>>): Prisma__DatasetClient<$Result.GetResult<Prisma.$DatasetPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first Dataset that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DatasetFindFirstArgs} args - Arguments to find a Dataset
     * @example
     * // Get one Dataset
     * const dataset = await prisma.dataset.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends DatasetFindFirstArgs>(args?: SelectSubset<T, DatasetFindFirstArgs<ExtArgs>>): Prisma__DatasetClient<$Result.GetResult<Prisma.$DatasetPayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first Dataset that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DatasetFindFirstOrThrowArgs} args - Arguments to find a Dataset
     * @example
     * // Get one Dataset
     * const dataset = await prisma.dataset.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends DatasetFindFirstOrThrowArgs>(args?: SelectSubset<T, DatasetFindFirstOrThrowArgs<ExtArgs>>): Prisma__DatasetClient<$Result.GetResult<Prisma.$DatasetPayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more Datasets that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DatasetFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all Datasets
     * const datasets = await prisma.dataset.findMany()
     * 
     * // Get first 10 Datasets
     * const datasets = await prisma.dataset.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const datasetWithIdOnly = await prisma.dataset.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends DatasetFindManyArgs>(args?: SelectSubset<T, DatasetFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$DatasetPayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a Dataset.
     * @param {DatasetCreateArgs} args - Arguments to create a Dataset.
     * @example
     * // Create one Dataset
     * const Dataset = await prisma.dataset.create({
     *   data: {
     *     // ... data to create a Dataset
     *   }
     * })
     * 
     */
    create<T extends DatasetCreateArgs>(args: SelectSubset<T, DatasetCreateArgs<ExtArgs>>): Prisma__DatasetClient<$Result.GetResult<Prisma.$DatasetPayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many Datasets.
     * @param {DatasetCreateManyArgs} args - Arguments to create many Datasets.
     * @example
     * // Create many Datasets
     * const dataset = await prisma.dataset.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends DatasetCreateManyArgs>(args?: SelectSubset<T, DatasetCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many Datasets and returns the data saved in the database.
     * @param {DatasetCreateManyAndReturnArgs} args - Arguments to create many Datasets.
     * @example
     * // Create many Datasets
     * const dataset = await prisma.dataset.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many Datasets and only return the `id`
     * const datasetWithIdOnly = await prisma.dataset.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends DatasetCreateManyAndReturnArgs>(args?: SelectSubset<T, DatasetCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$DatasetPayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a Dataset.
     * @param {DatasetDeleteArgs} args - Arguments to delete one Dataset.
     * @example
     * // Delete one Dataset
     * const Dataset = await prisma.dataset.delete({
     *   where: {
     *     // ... filter to delete one Dataset
     *   }
     * })
     * 
     */
    delete<T extends DatasetDeleteArgs>(args: SelectSubset<T, DatasetDeleteArgs<ExtArgs>>): Prisma__DatasetClient<$Result.GetResult<Prisma.$DatasetPayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one Dataset.
     * @param {DatasetUpdateArgs} args - Arguments to update one Dataset.
     * @example
     * // Update one Dataset
     * const dataset = await prisma.dataset.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends DatasetUpdateArgs>(args: SelectSubset<T, DatasetUpdateArgs<ExtArgs>>): Prisma__DatasetClient<$Result.GetResult<Prisma.$DatasetPayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more Datasets.
     * @param {DatasetDeleteManyArgs} args - Arguments to filter Datasets to delete.
     * @example
     * // Delete a few Datasets
     * const { count } = await prisma.dataset.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends DatasetDeleteManyArgs>(args?: SelectSubset<T, DatasetDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more Datasets.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DatasetUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many Datasets
     * const dataset = await prisma.dataset.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends DatasetUpdateManyArgs>(args: SelectSubset<T, DatasetUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create or update one Dataset.
     * @param {DatasetUpsertArgs} args - Arguments to update or create a Dataset.
     * @example
     * // Update or create a Dataset
     * const dataset = await prisma.dataset.upsert({
     *   create: {
     *     // ... data to create a Dataset
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the Dataset we want to update
     *   }
     * })
     */
    upsert<T extends DatasetUpsertArgs>(args: SelectSubset<T, DatasetUpsertArgs<ExtArgs>>): Prisma__DatasetClient<$Result.GetResult<Prisma.$DatasetPayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of Datasets.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DatasetCountArgs} args - Arguments to filter Datasets to count.
     * @example
     * // Count the number of Datasets
     * const count = await prisma.dataset.count({
     *   where: {
     *     // ... the filter for the Datasets we want to count
     *   }
     * })
    **/
    count<T extends DatasetCountArgs>(
      args?: Subset<T, DatasetCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], DatasetCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a Dataset.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DatasetAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends DatasetAggregateArgs>(args: Subset<T, DatasetAggregateArgs>): Prisma.PrismaPromise<GetDatasetAggregateType<T>>

    /**
     * Group by Dataset.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DatasetGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends DatasetGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: DatasetGroupByArgs['orderBy'] }
        : { orderBy?: DatasetGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, DatasetGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetDatasetGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the Dataset model
   */
  readonly fields: DatasetFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for Dataset.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__DatasetClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    user<T extends UserDefaultArgs<ExtArgs> = {}>(args?: Subset<T, UserDefaultArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions> | Null, Null, ExtArgs, GlobalOmitOptions>
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the Dataset model
   */
  interface DatasetFieldRefs {
    readonly id: FieldRef<"Dataset", 'String'>
    readonly userId: FieldRef<"Dataset", 'Int'>
    readonly filePath: FieldRef<"Dataset", 'String'>
    readonly name: FieldRef<"Dataset", 'String'>
    readonly description: FieldRef<"Dataset", 'String'>
    readonly rowCount: FieldRef<"Dataset", 'Int'>
    readonly columnCount: FieldRef<"Dataset", 'Int'>
    readonly createdAt: FieldRef<"Dataset", 'DateTime'>
    readonly updatedAt: FieldRef<"Dataset", 'DateTime'>
  }
    

  // Custom InputTypes
  /**
   * Dataset findUnique
   */
  export type DatasetFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Dataset
     */
    select?: DatasetSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Dataset
     */
    omit?: DatasetOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DatasetInclude<ExtArgs> | null
    /**
     * Filter, which Dataset to fetch.
     */
    where: DatasetWhereUniqueInput
  }

  /**
   * Dataset findUniqueOrThrow
   */
  export type DatasetFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Dataset
     */
    select?: DatasetSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Dataset
     */
    omit?: DatasetOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DatasetInclude<ExtArgs> | null
    /**
     * Filter, which Dataset to fetch.
     */
    where: DatasetWhereUniqueInput
  }

  /**
   * Dataset findFirst
   */
  export type DatasetFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Dataset
     */
    select?: DatasetSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Dataset
     */
    omit?: DatasetOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DatasetInclude<ExtArgs> | null
    /**
     * Filter, which Dataset to fetch.
     */
    where?: DatasetWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Datasets to fetch.
     */
    orderBy?: DatasetOrderByWithRelationInput | DatasetOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for Datasets.
     */
    cursor?: DatasetWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Datasets from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Datasets.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of Datasets.
     */
    distinct?: DatasetScalarFieldEnum | DatasetScalarFieldEnum[]
  }

  /**
   * Dataset findFirstOrThrow
   */
  export type DatasetFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Dataset
     */
    select?: DatasetSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Dataset
     */
    omit?: DatasetOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DatasetInclude<ExtArgs> | null
    /**
     * Filter, which Dataset to fetch.
     */
    where?: DatasetWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Datasets to fetch.
     */
    orderBy?: DatasetOrderByWithRelationInput | DatasetOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for Datasets.
     */
    cursor?: DatasetWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Datasets from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Datasets.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of Datasets.
     */
    distinct?: DatasetScalarFieldEnum | DatasetScalarFieldEnum[]
  }

  /**
   * Dataset findMany
   */
  export type DatasetFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Dataset
     */
    select?: DatasetSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Dataset
     */
    omit?: DatasetOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DatasetInclude<ExtArgs> | null
    /**
     * Filter, which Datasets to fetch.
     */
    where?: DatasetWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Datasets to fetch.
     */
    orderBy?: DatasetOrderByWithRelationInput | DatasetOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing Datasets.
     */
    cursor?: DatasetWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Datasets from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Datasets.
     */
    skip?: number
    distinct?: DatasetScalarFieldEnum | DatasetScalarFieldEnum[]
  }

  /**
   * Dataset create
   */
  export type DatasetCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Dataset
     */
    select?: DatasetSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Dataset
     */
    omit?: DatasetOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DatasetInclude<ExtArgs> | null
    /**
     * The data needed to create a Dataset.
     */
    data: XOR<DatasetCreateInput, DatasetUncheckedCreateInput>
  }

  /**
   * Dataset createMany
   */
  export type DatasetCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many Datasets.
     */
    data: DatasetCreateManyInput | DatasetCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * Dataset createManyAndReturn
   */
  export type DatasetCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Dataset
     */
    select?: DatasetSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the Dataset
     */
    omit?: DatasetOmit<ExtArgs> | null
    /**
     * The data used to create many Datasets.
     */
    data: DatasetCreateManyInput | DatasetCreateManyInput[]
    skipDuplicates?: boolean
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DatasetIncludeCreateManyAndReturn<ExtArgs> | null
  }

  /**
   * Dataset update
   */
  export type DatasetUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Dataset
     */
    select?: DatasetSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Dataset
     */
    omit?: DatasetOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DatasetInclude<ExtArgs> | null
    /**
     * The data needed to update a Dataset.
     */
    data: XOR<DatasetUpdateInput, DatasetUncheckedUpdateInput>
    /**
     * Choose, which Dataset to update.
     */
    where: DatasetWhereUniqueInput
  }

  /**
   * Dataset updateMany
   */
  export type DatasetUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update Datasets.
     */
    data: XOR<DatasetUpdateManyMutationInput, DatasetUncheckedUpdateManyInput>
    /**
     * Filter which Datasets to update
     */
    where?: DatasetWhereInput
  }

  /**
   * Dataset upsert
   */
  export type DatasetUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Dataset
     */
    select?: DatasetSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Dataset
     */
    omit?: DatasetOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DatasetInclude<ExtArgs> | null
    /**
     * The filter to search for the Dataset to update in case it exists.
     */
    where: DatasetWhereUniqueInput
    /**
     * In case the Dataset found by the `where` argument doesn't exist, create a new Dataset with this data.
     */
    create: XOR<DatasetCreateInput, DatasetUncheckedCreateInput>
    /**
     * In case the Dataset was found with the provided `where` argument, update it with this data.
     */
    update: XOR<DatasetUpdateInput, DatasetUncheckedUpdateInput>
  }

  /**
   * Dataset delete
   */
  export type DatasetDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Dataset
     */
    select?: DatasetSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Dataset
     */
    omit?: DatasetOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DatasetInclude<ExtArgs> | null
    /**
     * Filter which Dataset to delete.
     */
    where: DatasetWhereUniqueInput
  }

  /**
   * Dataset deleteMany
   */
  export type DatasetDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which Datasets to delete
     */
    where?: DatasetWhereInput
  }

  /**
   * Dataset without action
   */
  export type DatasetDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Dataset
     */
    select?: DatasetSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Dataset
     */
    omit?: DatasetOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DatasetInclude<ExtArgs> | null
  }


  /**
   * Enums
   */

  export const TransactionIsolationLevel: {
    ReadUncommitted: 'ReadUncommitted',
    ReadCommitted: 'ReadCommitted',
    RepeatableRead: 'RepeatableRead',
    Serializable: 'Serializable'
  };

  export type TransactionIsolationLevel = (typeof TransactionIsolationLevel)[keyof typeof TransactionIsolationLevel]


  export const UserScalarFieldEnum: {
    id: 'id',
    email: 'email',
    name: 'name',
    degree: 'degree',
    field: 'field',
    university: 'university',
    graduationYear: 'graduationYear',
    password: 'password',
    createdAt: 'createdAt',
    updatedAt: 'updatedAt'
  };

  export type UserScalarFieldEnum = (typeof UserScalarFieldEnum)[keyof typeof UserScalarFieldEnum]


  export const TrainingRunScalarFieldEnum: {
    id: 'id',
    userId: 'userId',
    name: 'name',
    description: 'description',
    status: 'status',
    datasetUri: 'datasetUri',
    modelPath: 'modelPath',
    resultsPath: 'resultsPath',
    metrics: 'metrics',
    parameters: 'parameters',
    geneDepmapResults: 'geneDepmapResults',
    createdAt: 'createdAt',
    updatedAt: 'updatedAt'
  };

  export type TrainingRunScalarFieldEnum = (typeof TrainingRunScalarFieldEnum)[keyof typeof TrainingRunScalarFieldEnum]


  export const DatasetScalarFieldEnum: {
    id: 'id',
    userId: 'userId',
    filePath: 'filePath',
    name: 'name',
    description: 'description',
    rowCount: 'rowCount',
    columnCount: 'columnCount',
    createdAt: 'createdAt',
    updatedAt: 'updatedAt'
  };

  export type DatasetScalarFieldEnum = (typeof DatasetScalarFieldEnum)[keyof typeof DatasetScalarFieldEnum]


  export const SortOrder: {
    asc: 'asc',
    desc: 'desc'
  };

  export type SortOrder = (typeof SortOrder)[keyof typeof SortOrder]


  export const NullableJsonNullValueInput: {
    DbNull: typeof DbNull,
    JsonNull: typeof JsonNull
  };

  export type NullableJsonNullValueInput = (typeof NullableJsonNullValueInput)[keyof typeof NullableJsonNullValueInput]


  export const QueryMode: {
    default: 'default',
    insensitive: 'insensitive'
  };

  export type QueryMode = (typeof QueryMode)[keyof typeof QueryMode]


  export const JsonNullValueFilter: {
    DbNull: typeof DbNull,
    JsonNull: typeof JsonNull,
    AnyNull: typeof AnyNull
  };

  export type JsonNullValueFilter = (typeof JsonNullValueFilter)[keyof typeof JsonNullValueFilter]


  export const NullsOrder: {
    first: 'first',
    last: 'last'
  };

  export type NullsOrder = (typeof NullsOrder)[keyof typeof NullsOrder]


  /**
   * Field references
   */


  /**
   * Reference to a field of type 'Int'
   */
  export type IntFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Int'>
    


  /**
   * Reference to a field of type 'Int[]'
   */
  export type ListIntFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Int[]'>
    


  /**
   * Reference to a field of type 'String'
   */
  export type StringFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'String'>
    


  /**
   * Reference to a field of type 'String[]'
   */
  export type ListStringFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'String[]'>
    


  /**
   * Reference to a field of type 'DateTime'
   */
  export type DateTimeFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'DateTime'>
    


  /**
   * Reference to a field of type 'DateTime[]'
   */
  export type ListDateTimeFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'DateTime[]'>
    


  /**
   * Reference to a field of type 'Json'
   */
  export type JsonFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Json'>
    


  /**
   * Reference to a field of type 'Float'
   */
  export type FloatFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Float'>
    


  /**
   * Reference to a field of type 'Float[]'
   */
  export type ListFloatFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Float[]'>
    
  /**
   * Deep Input Types
   */


  export type UserWhereInput = {
    AND?: UserWhereInput | UserWhereInput[]
    OR?: UserWhereInput[]
    NOT?: UserWhereInput | UserWhereInput[]
    id?: IntFilter<"User"> | number
    email?: StringFilter<"User"> | string
    name?: StringFilter<"User"> | string
    degree?: StringFilter<"User"> | string
    field?: StringFilter<"User"> | string
    university?: StringFilter<"User"> | string
    graduationYear?: IntFilter<"User"> | number
    password?: StringFilter<"User"> | string
    createdAt?: DateTimeFilter<"User"> | Date | string
    updatedAt?: DateTimeFilter<"User"> | Date | string
    runs?: TrainingRunListRelationFilter
    datasets?: DatasetListRelationFilter
  }

  export type UserOrderByWithRelationInput = {
    id?: SortOrder
    email?: SortOrder
    name?: SortOrder
    degree?: SortOrder
    field?: SortOrder
    university?: SortOrder
    graduationYear?: SortOrder
    password?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    runs?: TrainingRunOrderByRelationAggregateInput
    datasets?: DatasetOrderByRelationAggregateInput
  }

  export type UserWhereUniqueInput = Prisma.AtLeast<{
    id?: number
    email?: string
    AND?: UserWhereInput | UserWhereInput[]
    OR?: UserWhereInput[]
    NOT?: UserWhereInput | UserWhereInput[]
    name?: StringFilter<"User"> | string
    degree?: StringFilter<"User"> | string
    field?: StringFilter<"User"> | string
    university?: StringFilter<"User"> | string
    graduationYear?: IntFilter<"User"> | number
    password?: StringFilter<"User"> | string
    createdAt?: DateTimeFilter<"User"> | Date | string
    updatedAt?: DateTimeFilter<"User"> | Date | string
    runs?: TrainingRunListRelationFilter
    datasets?: DatasetListRelationFilter
  }, "id" | "email">

  export type UserOrderByWithAggregationInput = {
    id?: SortOrder
    email?: SortOrder
    name?: SortOrder
    degree?: SortOrder
    field?: SortOrder
    university?: SortOrder
    graduationYear?: SortOrder
    password?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    _count?: UserCountOrderByAggregateInput
    _avg?: UserAvgOrderByAggregateInput
    _max?: UserMaxOrderByAggregateInput
    _min?: UserMinOrderByAggregateInput
    _sum?: UserSumOrderByAggregateInput
  }

  export type UserScalarWhereWithAggregatesInput = {
    AND?: UserScalarWhereWithAggregatesInput | UserScalarWhereWithAggregatesInput[]
    OR?: UserScalarWhereWithAggregatesInput[]
    NOT?: UserScalarWhereWithAggregatesInput | UserScalarWhereWithAggregatesInput[]
    id?: IntWithAggregatesFilter<"User"> | number
    email?: StringWithAggregatesFilter<"User"> | string
    name?: StringWithAggregatesFilter<"User"> | string
    degree?: StringWithAggregatesFilter<"User"> | string
    field?: StringWithAggregatesFilter<"User"> | string
    university?: StringWithAggregatesFilter<"User"> | string
    graduationYear?: IntWithAggregatesFilter<"User"> | number
    password?: StringWithAggregatesFilter<"User"> | string
    createdAt?: DateTimeWithAggregatesFilter<"User"> | Date | string
    updatedAt?: DateTimeWithAggregatesFilter<"User"> | Date | string
  }

  export type TrainingRunWhereInput = {
    AND?: TrainingRunWhereInput | TrainingRunWhereInput[]
    OR?: TrainingRunWhereInput[]
    NOT?: TrainingRunWhereInput | TrainingRunWhereInput[]
    id?: StringFilter<"TrainingRun"> | string
    userId?: IntFilter<"TrainingRun"> | number
    name?: StringFilter<"TrainingRun"> | string
    description?: StringFilter<"TrainingRun"> | string
    status?: StringFilter<"TrainingRun"> | string
    datasetUri?: StringFilter<"TrainingRun"> | string
    modelPath?: StringNullableFilter<"TrainingRun"> | string | null
    resultsPath?: StringNullableFilter<"TrainingRun"> | string | null
    metrics?: JsonNullableFilter<"TrainingRun">
    parameters?: JsonNullableFilter<"TrainingRun">
    geneDepmapResults?: JsonNullableFilter<"TrainingRun">
    createdAt?: DateTimeFilter<"TrainingRun"> | Date | string
    updatedAt?: DateTimeFilter<"TrainingRun"> | Date | string
    user?: XOR<UserRelationFilter, UserWhereInput>
  }

  export type TrainingRunOrderByWithRelationInput = {
    id?: SortOrder
    userId?: SortOrder
    name?: SortOrder
    description?: SortOrder
    status?: SortOrder
    datasetUri?: SortOrder
    modelPath?: SortOrderInput | SortOrder
    resultsPath?: SortOrderInput | SortOrder
    metrics?: SortOrderInput | SortOrder
    parameters?: SortOrderInput | SortOrder
    geneDepmapResults?: SortOrderInput | SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    user?: UserOrderByWithRelationInput
  }

  export type TrainingRunWhereUniqueInput = Prisma.AtLeast<{
    id?: string
    AND?: TrainingRunWhereInput | TrainingRunWhereInput[]
    OR?: TrainingRunWhereInput[]
    NOT?: TrainingRunWhereInput | TrainingRunWhereInput[]
    userId?: IntFilter<"TrainingRun"> | number
    name?: StringFilter<"TrainingRun"> | string
    description?: StringFilter<"TrainingRun"> | string
    status?: StringFilter<"TrainingRun"> | string
    datasetUri?: StringFilter<"TrainingRun"> | string
    modelPath?: StringNullableFilter<"TrainingRun"> | string | null
    resultsPath?: StringNullableFilter<"TrainingRun"> | string | null
    metrics?: JsonNullableFilter<"TrainingRun">
    parameters?: JsonNullableFilter<"TrainingRun">
    geneDepmapResults?: JsonNullableFilter<"TrainingRun">
    createdAt?: DateTimeFilter<"TrainingRun"> | Date | string
    updatedAt?: DateTimeFilter<"TrainingRun"> | Date | string
    user?: XOR<UserRelationFilter, UserWhereInput>
  }, "id">

  export type TrainingRunOrderByWithAggregationInput = {
    id?: SortOrder
    userId?: SortOrder
    name?: SortOrder
    description?: SortOrder
    status?: SortOrder
    datasetUri?: SortOrder
    modelPath?: SortOrderInput | SortOrder
    resultsPath?: SortOrderInput | SortOrder
    metrics?: SortOrderInput | SortOrder
    parameters?: SortOrderInput | SortOrder
    geneDepmapResults?: SortOrderInput | SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    _count?: TrainingRunCountOrderByAggregateInput
    _avg?: TrainingRunAvgOrderByAggregateInput
    _max?: TrainingRunMaxOrderByAggregateInput
    _min?: TrainingRunMinOrderByAggregateInput
    _sum?: TrainingRunSumOrderByAggregateInput
  }

  export type TrainingRunScalarWhereWithAggregatesInput = {
    AND?: TrainingRunScalarWhereWithAggregatesInput | TrainingRunScalarWhereWithAggregatesInput[]
    OR?: TrainingRunScalarWhereWithAggregatesInput[]
    NOT?: TrainingRunScalarWhereWithAggregatesInput | TrainingRunScalarWhereWithAggregatesInput[]
    id?: StringWithAggregatesFilter<"TrainingRun"> | string
    userId?: IntWithAggregatesFilter<"TrainingRun"> | number
    name?: StringWithAggregatesFilter<"TrainingRun"> | string
    description?: StringWithAggregatesFilter<"TrainingRun"> | string
    status?: StringWithAggregatesFilter<"TrainingRun"> | string
    datasetUri?: StringWithAggregatesFilter<"TrainingRun"> | string
    modelPath?: StringNullableWithAggregatesFilter<"TrainingRun"> | string | null
    resultsPath?: StringNullableWithAggregatesFilter<"TrainingRun"> | string | null
    metrics?: JsonNullableWithAggregatesFilter<"TrainingRun">
    parameters?: JsonNullableWithAggregatesFilter<"TrainingRun">
    geneDepmapResults?: JsonNullableWithAggregatesFilter<"TrainingRun">
    createdAt?: DateTimeWithAggregatesFilter<"TrainingRun"> | Date | string
    updatedAt?: DateTimeWithAggregatesFilter<"TrainingRun"> | Date | string
  }

  export type DatasetWhereInput = {
    AND?: DatasetWhereInput | DatasetWhereInput[]
    OR?: DatasetWhereInput[]
    NOT?: DatasetWhereInput | DatasetWhereInput[]
    id?: StringFilter<"Dataset"> | string
    userId?: IntFilter<"Dataset"> | number
    filePath?: StringFilter<"Dataset"> | string
    name?: StringFilter<"Dataset"> | string
    description?: StringFilter<"Dataset"> | string
    rowCount?: IntNullableFilter<"Dataset"> | number | null
    columnCount?: IntNullableFilter<"Dataset"> | number | null
    createdAt?: DateTimeFilter<"Dataset"> | Date | string
    updatedAt?: DateTimeFilter<"Dataset"> | Date | string
    user?: XOR<UserRelationFilter, UserWhereInput>
  }

  export type DatasetOrderByWithRelationInput = {
    id?: SortOrder
    userId?: SortOrder
    filePath?: SortOrder
    name?: SortOrder
    description?: SortOrder
    rowCount?: SortOrderInput | SortOrder
    columnCount?: SortOrderInput | SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    user?: UserOrderByWithRelationInput
  }

  export type DatasetWhereUniqueInput = Prisma.AtLeast<{
    id?: string
    AND?: DatasetWhereInput | DatasetWhereInput[]
    OR?: DatasetWhereInput[]
    NOT?: DatasetWhereInput | DatasetWhereInput[]
    userId?: IntFilter<"Dataset"> | number
    filePath?: StringFilter<"Dataset"> | string
    name?: StringFilter<"Dataset"> | string
    description?: StringFilter<"Dataset"> | string
    rowCount?: IntNullableFilter<"Dataset"> | number | null
    columnCount?: IntNullableFilter<"Dataset"> | number | null
    createdAt?: DateTimeFilter<"Dataset"> | Date | string
    updatedAt?: DateTimeFilter<"Dataset"> | Date | string
    user?: XOR<UserRelationFilter, UserWhereInput>
  }, "id">

  export type DatasetOrderByWithAggregationInput = {
    id?: SortOrder
    userId?: SortOrder
    filePath?: SortOrder
    name?: SortOrder
    description?: SortOrder
    rowCount?: SortOrderInput | SortOrder
    columnCount?: SortOrderInput | SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    _count?: DatasetCountOrderByAggregateInput
    _avg?: DatasetAvgOrderByAggregateInput
    _max?: DatasetMaxOrderByAggregateInput
    _min?: DatasetMinOrderByAggregateInput
    _sum?: DatasetSumOrderByAggregateInput
  }

  export type DatasetScalarWhereWithAggregatesInput = {
    AND?: DatasetScalarWhereWithAggregatesInput | DatasetScalarWhereWithAggregatesInput[]
    OR?: DatasetScalarWhereWithAggregatesInput[]
    NOT?: DatasetScalarWhereWithAggregatesInput | DatasetScalarWhereWithAggregatesInput[]
    id?: StringWithAggregatesFilter<"Dataset"> | string
    userId?: IntWithAggregatesFilter<"Dataset"> | number
    filePath?: StringWithAggregatesFilter<"Dataset"> | string
    name?: StringWithAggregatesFilter<"Dataset"> | string
    description?: StringWithAggregatesFilter<"Dataset"> | string
    rowCount?: IntNullableWithAggregatesFilter<"Dataset"> | number | null
    columnCount?: IntNullableWithAggregatesFilter<"Dataset"> | number | null
    createdAt?: DateTimeWithAggregatesFilter<"Dataset"> | Date | string
    updatedAt?: DateTimeWithAggregatesFilter<"Dataset"> | Date | string
  }

  export type UserCreateInput = {
    email: string
    name: string
    degree: string
    field: string
    university: string
    graduationYear: number
    password: string
    createdAt?: Date | string
    updatedAt?: Date | string
    runs?: TrainingRunCreateNestedManyWithoutUserInput
    datasets?: DatasetCreateNestedManyWithoutUserInput
  }

  export type UserUncheckedCreateInput = {
    id?: number
    email: string
    name: string
    degree: string
    field: string
    university: string
    graduationYear: number
    password: string
    createdAt?: Date | string
    updatedAt?: Date | string
    runs?: TrainingRunUncheckedCreateNestedManyWithoutUserInput
    datasets?: DatasetUncheckedCreateNestedManyWithoutUserInput
  }

  export type UserUpdateInput = {
    email?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    degree?: StringFieldUpdateOperationsInput | string
    field?: StringFieldUpdateOperationsInput | string
    university?: StringFieldUpdateOperationsInput | string
    graduationYear?: IntFieldUpdateOperationsInput | number
    password?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    runs?: TrainingRunUpdateManyWithoutUserNestedInput
    datasets?: DatasetUpdateManyWithoutUserNestedInput
  }

  export type UserUncheckedUpdateInput = {
    id?: IntFieldUpdateOperationsInput | number
    email?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    degree?: StringFieldUpdateOperationsInput | string
    field?: StringFieldUpdateOperationsInput | string
    university?: StringFieldUpdateOperationsInput | string
    graduationYear?: IntFieldUpdateOperationsInput | number
    password?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    runs?: TrainingRunUncheckedUpdateManyWithoutUserNestedInput
    datasets?: DatasetUncheckedUpdateManyWithoutUserNestedInput
  }

  export type UserCreateManyInput = {
    id?: number
    email: string
    name: string
    degree: string
    field: string
    university: string
    graduationYear: number
    password: string
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type UserUpdateManyMutationInput = {
    email?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    degree?: StringFieldUpdateOperationsInput | string
    field?: StringFieldUpdateOperationsInput | string
    university?: StringFieldUpdateOperationsInput | string
    graduationYear?: IntFieldUpdateOperationsInput | number
    password?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type UserUncheckedUpdateManyInput = {
    id?: IntFieldUpdateOperationsInput | number
    email?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    degree?: StringFieldUpdateOperationsInput | string
    field?: StringFieldUpdateOperationsInput | string
    university?: StringFieldUpdateOperationsInput | string
    graduationYear?: IntFieldUpdateOperationsInput | number
    password?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type TrainingRunCreateInput = {
    id: string
    name?: string
    description?: string
    status: string
    datasetUri: string
    modelPath?: string | null
    resultsPath?: string | null
    metrics?: NullableJsonNullValueInput | InputJsonValue
    parameters?: NullableJsonNullValueInput | InputJsonValue
    geneDepmapResults?: NullableJsonNullValueInput | InputJsonValue
    createdAt?: Date | string
    updatedAt?: Date | string
    user: UserCreateNestedOneWithoutRunsInput
  }

  export type TrainingRunUncheckedCreateInput = {
    id: string
    userId: number
    name?: string
    description?: string
    status: string
    datasetUri: string
    modelPath?: string | null
    resultsPath?: string | null
    metrics?: NullableJsonNullValueInput | InputJsonValue
    parameters?: NullableJsonNullValueInput | InputJsonValue
    geneDepmapResults?: NullableJsonNullValueInput | InputJsonValue
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type TrainingRunUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: StringFieldUpdateOperationsInput | string
    status?: StringFieldUpdateOperationsInput | string
    datasetUri?: StringFieldUpdateOperationsInput | string
    modelPath?: NullableStringFieldUpdateOperationsInput | string | null
    resultsPath?: NullableStringFieldUpdateOperationsInput | string | null
    metrics?: NullableJsonNullValueInput | InputJsonValue
    parameters?: NullableJsonNullValueInput | InputJsonValue
    geneDepmapResults?: NullableJsonNullValueInput | InputJsonValue
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    user?: UserUpdateOneRequiredWithoutRunsNestedInput
  }

  export type TrainingRunUncheckedUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    userId?: IntFieldUpdateOperationsInput | number
    name?: StringFieldUpdateOperationsInput | string
    description?: StringFieldUpdateOperationsInput | string
    status?: StringFieldUpdateOperationsInput | string
    datasetUri?: StringFieldUpdateOperationsInput | string
    modelPath?: NullableStringFieldUpdateOperationsInput | string | null
    resultsPath?: NullableStringFieldUpdateOperationsInput | string | null
    metrics?: NullableJsonNullValueInput | InputJsonValue
    parameters?: NullableJsonNullValueInput | InputJsonValue
    geneDepmapResults?: NullableJsonNullValueInput | InputJsonValue
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type TrainingRunCreateManyInput = {
    id: string
    userId: number
    name?: string
    description?: string
    status: string
    datasetUri: string
    modelPath?: string | null
    resultsPath?: string | null
    metrics?: NullableJsonNullValueInput | InputJsonValue
    parameters?: NullableJsonNullValueInput | InputJsonValue
    geneDepmapResults?: NullableJsonNullValueInput | InputJsonValue
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type TrainingRunUpdateManyMutationInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: StringFieldUpdateOperationsInput | string
    status?: StringFieldUpdateOperationsInput | string
    datasetUri?: StringFieldUpdateOperationsInput | string
    modelPath?: NullableStringFieldUpdateOperationsInput | string | null
    resultsPath?: NullableStringFieldUpdateOperationsInput | string | null
    metrics?: NullableJsonNullValueInput | InputJsonValue
    parameters?: NullableJsonNullValueInput | InputJsonValue
    geneDepmapResults?: NullableJsonNullValueInput | InputJsonValue
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type TrainingRunUncheckedUpdateManyInput = {
    id?: StringFieldUpdateOperationsInput | string
    userId?: IntFieldUpdateOperationsInput | number
    name?: StringFieldUpdateOperationsInput | string
    description?: StringFieldUpdateOperationsInput | string
    status?: StringFieldUpdateOperationsInput | string
    datasetUri?: StringFieldUpdateOperationsInput | string
    modelPath?: NullableStringFieldUpdateOperationsInput | string | null
    resultsPath?: NullableStringFieldUpdateOperationsInput | string | null
    metrics?: NullableJsonNullValueInput | InputJsonValue
    parameters?: NullableJsonNullValueInput | InputJsonValue
    geneDepmapResults?: NullableJsonNullValueInput | InputJsonValue
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type DatasetCreateInput = {
    id: string
    filePath: string
    name: string
    description: string
    rowCount?: number | null
    columnCount?: number | null
    createdAt?: Date | string
    updatedAt?: Date | string
    user: UserCreateNestedOneWithoutDatasetsInput
  }

  export type DatasetUncheckedCreateInput = {
    id: string
    userId: number
    filePath: string
    name: string
    description: string
    rowCount?: number | null
    columnCount?: number | null
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type DatasetUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    filePath?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: StringFieldUpdateOperationsInput | string
    rowCount?: NullableIntFieldUpdateOperationsInput | number | null
    columnCount?: NullableIntFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    user?: UserUpdateOneRequiredWithoutDatasetsNestedInput
  }

  export type DatasetUncheckedUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    userId?: IntFieldUpdateOperationsInput | number
    filePath?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: StringFieldUpdateOperationsInput | string
    rowCount?: NullableIntFieldUpdateOperationsInput | number | null
    columnCount?: NullableIntFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type DatasetCreateManyInput = {
    id: string
    userId: number
    filePath: string
    name: string
    description: string
    rowCount?: number | null
    columnCount?: number | null
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type DatasetUpdateManyMutationInput = {
    id?: StringFieldUpdateOperationsInput | string
    filePath?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: StringFieldUpdateOperationsInput | string
    rowCount?: NullableIntFieldUpdateOperationsInput | number | null
    columnCount?: NullableIntFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type DatasetUncheckedUpdateManyInput = {
    id?: StringFieldUpdateOperationsInput | string
    userId?: IntFieldUpdateOperationsInput | number
    filePath?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: StringFieldUpdateOperationsInput | string
    rowCount?: NullableIntFieldUpdateOperationsInput | number | null
    columnCount?: NullableIntFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type IntFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel>
    in?: number[] | ListIntFieldRefInput<$PrismaModel>
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel>
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntFilter<$PrismaModel> | number
  }

  export type StringFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel>
    in?: string[] | ListStringFieldRefInput<$PrismaModel>
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel>
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    mode?: QueryMode
    not?: NestedStringFilter<$PrismaModel> | string
  }

  export type DateTimeFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeFilter<$PrismaModel> | Date | string
  }

  export type TrainingRunListRelationFilter = {
    every?: TrainingRunWhereInput
    some?: TrainingRunWhereInput
    none?: TrainingRunWhereInput
  }

  export type DatasetListRelationFilter = {
    every?: DatasetWhereInput
    some?: DatasetWhereInput
    none?: DatasetWhereInput
  }

  export type TrainingRunOrderByRelationAggregateInput = {
    _count?: SortOrder
  }

  export type DatasetOrderByRelationAggregateInput = {
    _count?: SortOrder
  }

  export type UserCountOrderByAggregateInput = {
    id?: SortOrder
    email?: SortOrder
    name?: SortOrder
    degree?: SortOrder
    field?: SortOrder
    university?: SortOrder
    graduationYear?: SortOrder
    password?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type UserAvgOrderByAggregateInput = {
    id?: SortOrder
    graduationYear?: SortOrder
  }

  export type UserMaxOrderByAggregateInput = {
    id?: SortOrder
    email?: SortOrder
    name?: SortOrder
    degree?: SortOrder
    field?: SortOrder
    university?: SortOrder
    graduationYear?: SortOrder
    password?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type UserMinOrderByAggregateInput = {
    id?: SortOrder
    email?: SortOrder
    name?: SortOrder
    degree?: SortOrder
    field?: SortOrder
    university?: SortOrder
    graduationYear?: SortOrder
    password?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type UserSumOrderByAggregateInput = {
    id?: SortOrder
    graduationYear?: SortOrder
  }

  export type IntWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel>
    in?: number[] | ListIntFieldRefInput<$PrismaModel>
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel>
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntWithAggregatesFilter<$PrismaModel> | number
    _count?: NestedIntFilter<$PrismaModel>
    _avg?: NestedFloatFilter<$PrismaModel>
    _sum?: NestedIntFilter<$PrismaModel>
    _min?: NestedIntFilter<$PrismaModel>
    _max?: NestedIntFilter<$PrismaModel>
  }

  export type StringWithAggregatesFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel>
    in?: string[] | ListStringFieldRefInput<$PrismaModel>
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel>
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    mode?: QueryMode
    not?: NestedStringWithAggregatesFilter<$PrismaModel> | string
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedStringFilter<$PrismaModel>
    _max?: NestedStringFilter<$PrismaModel>
  }

  export type DateTimeWithAggregatesFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeWithAggregatesFilter<$PrismaModel> | Date | string
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedDateTimeFilter<$PrismaModel>
    _max?: NestedDateTimeFilter<$PrismaModel>
  }

  export type StringNullableFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel> | null
    in?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    mode?: QueryMode
    not?: NestedStringNullableFilter<$PrismaModel> | string | null
  }
  export type JsonNullableFilter<$PrismaModel = never> =
    | PatchUndefined<
        Either<Required<JsonNullableFilterBase<$PrismaModel>>, Exclude<keyof Required<JsonNullableFilterBase<$PrismaModel>>, 'path'>>,
        Required<JsonNullableFilterBase<$PrismaModel>>
      >
    | OptionalFlat<Omit<Required<JsonNullableFilterBase<$PrismaModel>>, 'path'>>

  export type JsonNullableFilterBase<$PrismaModel = never> = {
    equals?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | JsonNullValueFilter
    path?: string[]
    string_contains?: string | StringFieldRefInput<$PrismaModel>
    string_starts_with?: string | StringFieldRefInput<$PrismaModel>
    string_ends_with?: string | StringFieldRefInput<$PrismaModel>
    array_contains?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    array_starts_with?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    array_ends_with?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    lt?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    lte?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    gt?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    gte?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    not?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | JsonNullValueFilter
  }

  export type UserRelationFilter = {
    is?: UserWhereInput
    isNot?: UserWhereInput
  }

  export type SortOrderInput = {
    sort: SortOrder
    nulls?: NullsOrder
  }

  export type TrainingRunCountOrderByAggregateInput = {
    id?: SortOrder
    userId?: SortOrder
    name?: SortOrder
    description?: SortOrder
    status?: SortOrder
    datasetUri?: SortOrder
    modelPath?: SortOrder
    resultsPath?: SortOrder
    metrics?: SortOrder
    parameters?: SortOrder
    geneDepmapResults?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type TrainingRunAvgOrderByAggregateInput = {
    userId?: SortOrder
  }

  export type TrainingRunMaxOrderByAggregateInput = {
    id?: SortOrder
    userId?: SortOrder
    name?: SortOrder
    description?: SortOrder
    status?: SortOrder
    datasetUri?: SortOrder
    modelPath?: SortOrder
    resultsPath?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type TrainingRunMinOrderByAggregateInput = {
    id?: SortOrder
    userId?: SortOrder
    name?: SortOrder
    description?: SortOrder
    status?: SortOrder
    datasetUri?: SortOrder
    modelPath?: SortOrder
    resultsPath?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type TrainingRunSumOrderByAggregateInput = {
    userId?: SortOrder
  }

  export type StringNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel> | null
    in?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    mode?: QueryMode
    not?: NestedStringNullableWithAggregatesFilter<$PrismaModel> | string | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _min?: NestedStringNullableFilter<$PrismaModel>
    _max?: NestedStringNullableFilter<$PrismaModel>
  }
  export type JsonNullableWithAggregatesFilter<$PrismaModel = never> =
    | PatchUndefined<
        Either<Required<JsonNullableWithAggregatesFilterBase<$PrismaModel>>, Exclude<keyof Required<JsonNullableWithAggregatesFilterBase<$PrismaModel>>, 'path'>>,
        Required<JsonNullableWithAggregatesFilterBase<$PrismaModel>>
      >
    | OptionalFlat<Omit<Required<JsonNullableWithAggregatesFilterBase<$PrismaModel>>, 'path'>>

  export type JsonNullableWithAggregatesFilterBase<$PrismaModel = never> = {
    equals?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | JsonNullValueFilter
    path?: string[]
    string_contains?: string | StringFieldRefInput<$PrismaModel>
    string_starts_with?: string | StringFieldRefInput<$PrismaModel>
    string_ends_with?: string | StringFieldRefInput<$PrismaModel>
    array_contains?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    array_starts_with?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    array_ends_with?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    lt?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    lte?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    gt?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    gte?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    not?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | JsonNullValueFilter
    _count?: NestedIntNullableFilter<$PrismaModel>
    _min?: NestedJsonNullableFilter<$PrismaModel>
    _max?: NestedJsonNullableFilter<$PrismaModel>
  }

  export type IntNullableFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel> | null
    in?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntNullableFilter<$PrismaModel> | number | null
  }

  export type DatasetCountOrderByAggregateInput = {
    id?: SortOrder
    userId?: SortOrder
    filePath?: SortOrder
    name?: SortOrder
    description?: SortOrder
    rowCount?: SortOrder
    columnCount?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type DatasetAvgOrderByAggregateInput = {
    userId?: SortOrder
    rowCount?: SortOrder
    columnCount?: SortOrder
  }

  export type DatasetMaxOrderByAggregateInput = {
    id?: SortOrder
    userId?: SortOrder
    filePath?: SortOrder
    name?: SortOrder
    description?: SortOrder
    rowCount?: SortOrder
    columnCount?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type DatasetMinOrderByAggregateInput = {
    id?: SortOrder
    userId?: SortOrder
    filePath?: SortOrder
    name?: SortOrder
    description?: SortOrder
    rowCount?: SortOrder
    columnCount?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type DatasetSumOrderByAggregateInput = {
    userId?: SortOrder
    rowCount?: SortOrder
    columnCount?: SortOrder
  }

  export type IntNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel> | null
    in?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntNullableWithAggregatesFilter<$PrismaModel> | number | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _avg?: NestedFloatNullableFilter<$PrismaModel>
    _sum?: NestedIntNullableFilter<$PrismaModel>
    _min?: NestedIntNullableFilter<$PrismaModel>
    _max?: NestedIntNullableFilter<$PrismaModel>
  }

  export type TrainingRunCreateNestedManyWithoutUserInput = {
    create?: XOR<TrainingRunCreateWithoutUserInput, TrainingRunUncheckedCreateWithoutUserInput> | TrainingRunCreateWithoutUserInput[] | TrainingRunUncheckedCreateWithoutUserInput[]
    connectOrCreate?: TrainingRunCreateOrConnectWithoutUserInput | TrainingRunCreateOrConnectWithoutUserInput[]
    createMany?: TrainingRunCreateManyUserInputEnvelope
    connect?: TrainingRunWhereUniqueInput | TrainingRunWhereUniqueInput[]
  }

  export type DatasetCreateNestedManyWithoutUserInput = {
    create?: XOR<DatasetCreateWithoutUserInput, DatasetUncheckedCreateWithoutUserInput> | DatasetCreateWithoutUserInput[] | DatasetUncheckedCreateWithoutUserInput[]
    connectOrCreate?: DatasetCreateOrConnectWithoutUserInput | DatasetCreateOrConnectWithoutUserInput[]
    createMany?: DatasetCreateManyUserInputEnvelope
    connect?: DatasetWhereUniqueInput | DatasetWhereUniqueInput[]
  }

  export type TrainingRunUncheckedCreateNestedManyWithoutUserInput = {
    create?: XOR<TrainingRunCreateWithoutUserInput, TrainingRunUncheckedCreateWithoutUserInput> | TrainingRunCreateWithoutUserInput[] | TrainingRunUncheckedCreateWithoutUserInput[]
    connectOrCreate?: TrainingRunCreateOrConnectWithoutUserInput | TrainingRunCreateOrConnectWithoutUserInput[]
    createMany?: TrainingRunCreateManyUserInputEnvelope
    connect?: TrainingRunWhereUniqueInput | TrainingRunWhereUniqueInput[]
  }

  export type DatasetUncheckedCreateNestedManyWithoutUserInput = {
    create?: XOR<DatasetCreateWithoutUserInput, DatasetUncheckedCreateWithoutUserInput> | DatasetCreateWithoutUserInput[] | DatasetUncheckedCreateWithoutUserInput[]
    connectOrCreate?: DatasetCreateOrConnectWithoutUserInput | DatasetCreateOrConnectWithoutUserInput[]
    createMany?: DatasetCreateManyUserInputEnvelope
    connect?: DatasetWhereUniqueInput | DatasetWhereUniqueInput[]
  }

  export type StringFieldUpdateOperationsInput = {
    set?: string
  }

  export type IntFieldUpdateOperationsInput = {
    set?: number
    increment?: number
    decrement?: number
    multiply?: number
    divide?: number
  }

  export type DateTimeFieldUpdateOperationsInput = {
    set?: Date | string
  }

  export type TrainingRunUpdateManyWithoutUserNestedInput = {
    create?: XOR<TrainingRunCreateWithoutUserInput, TrainingRunUncheckedCreateWithoutUserInput> | TrainingRunCreateWithoutUserInput[] | TrainingRunUncheckedCreateWithoutUserInput[]
    connectOrCreate?: TrainingRunCreateOrConnectWithoutUserInput | TrainingRunCreateOrConnectWithoutUserInput[]
    upsert?: TrainingRunUpsertWithWhereUniqueWithoutUserInput | TrainingRunUpsertWithWhereUniqueWithoutUserInput[]
    createMany?: TrainingRunCreateManyUserInputEnvelope
    set?: TrainingRunWhereUniqueInput | TrainingRunWhereUniqueInput[]
    disconnect?: TrainingRunWhereUniqueInput | TrainingRunWhereUniqueInput[]
    delete?: TrainingRunWhereUniqueInput | TrainingRunWhereUniqueInput[]
    connect?: TrainingRunWhereUniqueInput | TrainingRunWhereUniqueInput[]
    update?: TrainingRunUpdateWithWhereUniqueWithoutUserInput | TrainingRunUpdateWithWhereUniqueWithoutUserInput[]
    updateMany?: TrainingRunUpdateManyWithWhereWithoutUserInput | TrainingRunUpdateManyWithWhereWithoutUserInput[]
    deleteMany?: TrainingRunScalarWhereInput | TrainingRunScalarWhereInput[]
  }

  export type DatasetUpdateManyWithoutUserNestedInput = {
    create?: XOR<DatasetCreateWithoutUserInput, DatasetUncheckedCreateWithoutUserInput> | DatasetCreateWithoutUserInput[] | DatasetUncheckedCreateWithoutUserInput[]
    connectOrCreate?: DatasetCreateOrConnectWithoutUserInput | DatasetCreateOrConnectWithoutUserInput[]
    upsert?: DatasetUpsertWithWhereUniqueWithoutUserInput | DatasetUpsertWithWhereUniqueWithoutUserInput[]
    createMany?: DatasetCreateManyUserInputEnvelope
    set?: DatasetWhereUniqueInput | DatasetWhereUniqueInput[]
    disconnect?: DatasetWhereUniqueInput | DatasetWhereUniqueInput[]
    delete?: DatasetWhereUniqueInput | DatasetWhereUniqueInput[]
    connect?: DatasetWhereUniqueInput | DatasetWhereUniqueInput[]
    update?: DatasetUpdateWithWhereUniqueWithoutUserInput | DatasetUpdateWithWhereUniqueWithoutUserInput[]
    updateMany?: DatasetUpdateManyWithWhereWithoutUserInput | DatasetUpdateManyWithWhereWithoutUserInput[]
    deleteMany?: DatasetScalarWhereInput | DatasetScalarWhereInput[]
  }

  export type TrainingRunUncheckedUpdateManyWithoutUserNestedInput = {
    create?: XOR<TrainingRunCreateWithoutUserInput, TrainingRunUncheckedCreateWithoutUserInput> | TrainingRunCreateWithoutUserInput[] | TrainingRunUncheckedCreateWithoutUserInput[]
    connectOrCreate?: TrainingRunCreateOrConnectWithoutUserInput | TrainingRunCreateOrConnectWithoutUserInput[]
    upsert?: TrainingRunUpsertWithWhereUniqueWithoutUserInput | TrainingRunUpsertWithWhereUniqueWithoutUserInput[]
    createMany?: TrainingRunCreateManyUserInputEnvelope
    set?: TrainingRunWhereUniqueInput | TrainingRunWhereUniqueInput[]
    disconnect?: TrainingRunWhereUniqueInput | TrainingRunWhereUniqueInput[]
    delete?: TrainingRunWhereUniqueInput | TrainingRunWhereUniqueInput[]
    connect?: TrainingRunWhereUniqueInput | TrainingRunWhereUniqueInput[]
    update?: TrainingRunUpdateWithWhereUniqueWithoutUserInput | TrainingRunUpdateWithWhereUniqueWithoutUserInput[]
    updateMany?: TrainingRunUpdateManyWithWhereWithoutUserInput | TrainingRunUpdateManyWithWhereWithoutUserInput[]
    deleteMany?: TrainingRunScalarWhereInput | TrainingRunScalarWhereInput[]
  }

  export type DatasetUncheckedUpdateManyWithoutUserNestedInput = {
    create?: XOR<DatasetCreateWithoutUserInput, DatasetUncheckedCreateWithoutUserInput> | DatasetCreateWithoutUserInput[] | DatasetUncheckedCreateWithoutUserInput[]
    connectOrCreate?: DatasetCreateOrConnectWithoutUserInput | DatasetCreateOrConnectWithoutUserInput[]
    upsert?: DatasetUpsertWithWhereUniqueWithoutUserInput | DatasetUpsertWithWhereUniqueWithoutUserInput[]
    createMany?: DatasetCreateManyUserInputEnvelope
    set?: DatasetWhereUniqueInput | DatasetWhereUniqueInput[]
    disconnect?: DatasetWhereUniqueInput | DatasetWhereUniqueInput[]
    delete?: DatasetWhereUniqueInput | DatasetWhereUniqueInput[]
    connect?: DatasetWhereUniqueInput | DatasetWhereUniqueInput[]
    update?: DatasetUpdateWithWhereUniqueWithoutUserInput | DatasetUpdateWithWhereUniqueWithoutUserInput[]
    updateMany?: DatasetUpdateManyWithWhereWithoutUserInput | DatasetUpdateManyWithWhereWithoutUserInput[]
    deleteMany?: DatasetScalarWhereInput | DatasetScalarWhereInput[]
  }

  export type UserCreateNestedOneWithoutRunsInput = {
    create?: XOR<UserCreateWithoutRunsInput, UserUncheckedCreateWithoutRunsInput>
    connectOrCreate?: UserCreateOrConnectWithoutRunsInput
    connect?: UserWhereUniqueInput
  }

  export type NullableStringFieldUpdateOperationsInput = {
    set?: string | null
  }

  export type UserUpdateOneRequiredWithoutRunsNestedInput = {
    create?: XOR<UserCreateWithoutRunsInput, UserUncheckedCreateWithoutRunsInput>
    connectOrCreate?: UserCreateOrConnectWithoutRunsInput
    upsert?: UserUpsertWithoutRunsInput
    connect?: UserWhereUniqueInput
    update?: XOR<XOR<UserUpdateToOneWithWhereWithoutRunsInput, UserUpdateWithoutRunsInput>, UserUncheckedUpdateWithoutRunsInput>
  }

  export type UserCreateNestedOneWithoutDatasetsInput = {
    create?: XOR<UserCreateWithoutDatasetsInput, UserUncheckedCreateWithoutDatasetsInput>
    connectOrCreate?: UserCreateOrConnectWithoutDatasetsInput
    connect?: UserWhereUniqueInput
  }

  export type NullableIntFieldUpdateOperationsInput = {
    set?: number | null
    increment?: number
    decrement?: number
    multiply?: number
    divide?: number
  }

  export type UserUpdateOneRequiredWithoutDatasetsNestedInput = {
    create?: XOR<UserCreateWithoutDatasetsInput, UserUncheckedCreateWithoutDatasetsInput>
    connectOrCreate?: UserCreateOrConnectWithoutDatasetsInput
    upsert?: UserUpsertWithoutDatasetsInput
    connect?: UserWhereUniqueInput
    update?: XOR<XOR<UserUpdateToOneWithWhereWithoutDatasetsInput, UserUpdateWithoutDatasetsInput>, UserUncheckedUpdateWithoutDatasetsInput>
  }

  export type NestedIntFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel>
    in?: number[] | ListIntFieldRefInput<$PrismaModel>
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel>
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntFilter<$PrismaModel> | number
  }

  export type NestedStringFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel>
    in?: string[] | ListStringFieldRefInput<$PrismaModel>
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel>
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    not?: NestedStringFilter<$PrismaModel> | string
  }

  export type NestedDateTimeFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeFilter<$PrismaModel> | Date | string
  }

  export type NestedIntWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel>
    in?: number[] | ListIntFieldRefInput<$PrismaModel>
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel>
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntWithAggregatesFilter<$PrismaModel> | number
    _count?: NestedIntFilter<$PrismaModel>
    _avg?: NestedFloatFilter<$PrismaModel>
    _sum?: NestedIntFilter<$PrismaModel>
    _min?: NestedIntFilter<$PrismaModel>
    _max?: NestedIntFilter<$PrismaModel>
  }

  export type NestedFloatFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel>
    in?: number[] | ListFloatFieldRefInput<$PrismaModel>
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel>
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatFilter<$PrismaModel> | number
  }

  export type NestedStringWithAggregatesFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel>
    in?: string[] | ListStringFieldRefInput<$PrismaModel>
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel>
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    not?: NestedStringWithAggregatesFilter<$PrismaModel> | string
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedStringFilter<$PrismaModel>
    _max?: NestedStringFilter<$PrismaModel>
  }

  export type NestedDateTimeWithAggregatesFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeWithAggregatesFilter<$PrismaModel> | Date | string
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedDateTimeFilter<$PrismaModel>
    _max?: NestedDateTimeFilter<$PrismaModel>
  }

  export type NestedStringNullableFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel> | null
    in?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    not?: NestedStringNullableFilter<$PrismaModel> | string | null
  }

  export type NestedStringNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel> | null
    in?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    not?: NestedStringNullableWithAggregatesFilter<$PrismaModel> | string | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _min?: NestedStringNullableFilter<$PrismaModel>
    _max?: NestedStringNullableFilter<$PrismaModel>
  }

  export type NestedIntNullableFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel> | null
    in?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntNullableFilter<$PrismaModel> | number | null
  }
  export type NestedJsonNullableFilter<$PrismaModel = never> =
    | PatchUndefined<
        Either<Required<NestedJsonNullableFilterBase<$PrismaModel>>, Exclude<keyof Required<NestedJsonNullableFilterBase<$PrismaModel>>, 'path'>>,
        Required<NestedJsonNullableFilterBase<$PrismaModel>>
      >
    | OptionalFlat<Omit<Required<NestedJsonNullableFilterBase<$PrismaModel>>, 'path'>>

  export type NestedJsonNullableFilterBase<$PrismaModel = never> = {
    equals?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | JsonNullValueFilter
    path?: string[]
    string_contains?: string | StringFieldRefInput<$PrismaModel>
    string_starts_with?: string | StringFieldRefInput<$PrismaModel>
    string_ends_with?: string | StringFieldRefInput<$PrismaModel>
    array_contains?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    array_starts_with?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    array_ends_with?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    lt?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    lte?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    gt?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    gte?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    not?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | JsonNullValueFilter
  }

  export type NestedIntNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel> | null
    in?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntNullableWithAggregatesFilter<$PrismaModel> | number | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _avg?: NestedFloatNullableFilter<$PrismaModel>
    _sum?: NestedIntNullableFilter<$PrismaModel>
    _min?: NestedIntNullableFilter<$PrismaModel>
    _max?: NestedIntNullableFilter<$PrismaModel>
  }

  export type NestedFloatNullableFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel> | null
    in?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatNullableFilter<$PrismaModel> | number | null
  }

  export type TrainingRunCreateWithoutUserInput = {
    id: string
    name?: string
    description?: string
    status: string
    datasetUri: string
    modelPath?: string | null
    resultsPath?: string | null
    metrics?: NullableJsonNullValueInput | InputJsonValue
    parameters?: NullableJsonNullValueInput | InputJsonValue
    geneDepmapResults?: NullableJsonNullValueInput | InputJsonValue
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type TrainingRunUncheckedCreateWithoutUserInput = {
    id: string
    name?: string
    description?: string
    status: string
    datasetUri: string
    modelPath?: string | null
    resultsPath?: string | null
    metrics?: NullableJsonNullValueInput | InputJsonValue
    parameters?: NullableJsonNullValueInput | InputJsonValue
    geneDepmapResults?: NullableJsonNullValueInput | InputJsonValue
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type TrainingRunCreateOrConnectWithoutUserInput = {
    where: TrainingRunWhereUniqueInput
    create: XOR<TrainingRunCreateWithoutUserInput, TrainingRunUncheckedCreateWithoutUserInput>
  }

  export type TrainingRunCreateManyUserInputEnvelope = {
    data: TrainingRunCreateManyUserInput | TrainingRunCreateManyUserInput[]
    skipDuplicates?: boolean
  }

  export type DatasetCreateWithoutUserInput = {
    id: string
    filePath: string
    name: string
    description: string
    rowCount?: number | null
    columnCount?: number | null
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type DatasetUncheckedCreateWithoutUserInput = {
    id: string
    filePath: string
    name: string
    description: string
    rowCount?: number | null
    columnCount?: number | null
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type DatasetCreateOrConnectWithoutUserInput = {
    where: DatasetWhereUniqueInput
    create: XOR<DatasetCreateWithoutUserInput, DatasetUncheckedCreateWithoutUserInput>
  }

  export type DatasetCreateManyUserInputEnvelope = {
    data: DatasetCreateManyUserInput | DatasetCreateManyUserInput[]
    skipDuplicates?: boolean
  }

  export type TrainingRunUpsertWithWhereUniqueWithoutUserInput = {
    where: TrainingRunWhereUniqueInput
    update: XOR<TrainingRunUpdateWithoutUserInput, TrainingRunUncheckedUpdateWithoutUserInput>
    create: XOR<TrainingRunCreateWithoutUserInput, TrainingRunUncheckedCreateWithoutUserInput>
  }

  export type TrainingRunUpdateWithWhereUniqueWithoutUserInput = {
    where: TrainingRunWhereUniqueInput
    data: XOR<TrainingRunUpdateWithoutUserInput, TrainingRunUncheckedUpdateWithoutUserInput>
  }

  export type TrainingRunUpdateManyWithWhereWithoutUserInput = {
    where: TrainingRunScalarWhereInput
    data: XOR<TrainingRunUpdateManyMutationInput, TrainingRunUncheckedUpdateManyWithoutUserInput>
  }

  export type TrainingRunScalarWhereInput = {
    AND?: TrainingRunScalarWhereInput | TrainingRunScalarWhereInput[]
    OR?: TrainingRunScalarWhereInput[]
    NOT?: TrainingRunScalarWhereInput | TrainingRunScalarWhereInput[]
    id?: StringFilter<"TrainingRun"> | string
    userId?: IntFilter<"TrainingRun"> | number
    name?: StringFilter<"TrainingRun"> | string
    description?: StringFilter<"TrainingRun"> | string
    status?: StringFilter<"TrainingRun"> | string
    datasetUri?: StringFilter<"TrainingRun"> | string
    modelPath?: StringNullableFilter<"TrainingRun"> | string | null
    resultsPath?: StringNullableFilter<"TrainingRun"> | string | null
    metrics?: JsonNullableFilter<"TrainingRun">
    parameters?: JsonNullableFilter<"TrainingRun">
    geneDepmapResults?: JsonNullableFilter<"TrainingRun">
    createdAt?: DateTimeFilter<"TrainingRun"> | Date | string
    updatedAt?: DateTimeFilter<"TrainingRun"> | Date | string
  }

  export type DatasetUpsertWithWhereUniqueWithoutUserInput = {
    where: DatasetWhereUniqueInput
    update: XOR<DatasetUpdateWithoutUserInput, DatasetUncheckedUpdateWithoutUserInput>
    create: XOR<DatasetCreateWithoutUserInput, DatasetUncheckedCreateWithoutUserInput>
  }

  export type DatasetUpdateWithWhereUniqueWithoutUserInput = {
    where: DatasetWhereUniqueInput
    data: XOR<DatasetUpdateWithoutUserInput, DatasetUncheckedUpdateWithoutUserInput>
  }

  export type DatasetUpdateManyWithWhereWithoutUserInput = {
    where: DatasetScalarWhereInput
    data: XOR<DatasetUpdateManyMutationInput, DatasetUncheckedUpdateManyWithoutUserInput>
  }

  export type DatasetScalarWhereInput = {
    AND?: DatasetScalarWhereInput | DatasetScalarWhereInput[]
    OR?: DatasetScalarWhereInput[]
    NOT?: DatasetScalarWhereInput | DatasetScalarWhereInput[]
    id?: StringFilter<"Dataset"> | string
    userId?: IntFilter<"Dataset"> | number
    filePath?: StringFilter<"Dataset"> | string
    name?: StringFilter<"Dataset"> | string
    description?: StringFilter<"Dataset"> | string
    rowCount?: IntNullableFilter<"Dataset"> | number | null
    columnCount?: IntNullableFilter<"Dataset"> | number | null
    createdAt?: DateTimeFilter<"Dataset"> | Date | string
    updatedAt?: DateTimeFilter<"Dataset"> | Date | string
  }

  export type UserCreateWithoutRunsInput = {
    email: string
    name: string
    degree: string
    field: string
    university: string
    graduationYear: number
    password: string
    createdAt?: Date | string
    updatedAt?: Date | string
    datasets?: DatasetCreateNestedManyWithoutUserInput
  }

  export type UserUncheckedCreateWithoutRunsInput = {
    id?: number
    email: string
    name: string
    degree: string
    field: string
    university: string
    graduationYear: number
    password: string
    createdAt?: Date | string
    updatedAt?: Date | string
    datasets?: DatasetUncheckedCreateNestedManyWithoutUserInput
  }

  export type UserCreateOrConnectWithoutRunsInput = {
    where: UserWhereUniqueInput
    create: XOR<UserCreateWithoutRunsInput, UserUncheckedCreateWithoutRunsInput>
  }

  export type UserUpsertWithoutRunsInput = {
    update: XOR<UserUpdateWithoutRunsInput, UserUncheckedUpdateWithoutRunsInput>
    create: XOR<UserCreateWithoutRunsInput, UserUncheckedCreateWithoutRunsInput>
    where?: UserWhereInput
  }

  export type UserUpdateToOneWithWhereWithoutRunsInput = {
    where?: UserWhereInput
    data: XOR<UserUpdateWithoutRunsInput, UserUncheckedUpdateWithoutRunsInput>
  }

  export type UserUpdateWithoutRunsInput = {
    email?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    degree?: StringFieldUpdateOperationsInput | string
    field?: StringFieldUpdateOperationsInput | string
    university?: StringFieldUpdateOperationsInput | string
    graduationYear?: IntFieldUpdateOperationsInput | number
    password?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    datasets?: DatasetUpdateManyWithoutUserNestedInput
  }

  export type UserUncheckedUpdateWithoutRunsInput = {
    id?: IntFieldUpdateOperationsInput | number
    email?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    degree?: StringFieldUpdateOperationsInput | string
    field?: StringFieldUpdateOperationsInput | string
    university?: StringFieldUpdateOperationsInput | string
    graduationYear?: IntFieldUpdateOperationsInput | number
    password?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    datasets?: DatasetUncheckedUpdateManyWithoutUserNestedInput
  }

  export type UserCreateWithoutDatasetsInput = {
    email: string
    name: string
    degree: string
    field: string
    university: string
    graduationYear: number
    password: string
    createdAt?: Date | string
    updatedAt?: Date | string
    runs?: TrainingRunCreateNestedManyWithoutUserInput
  }

  export type UserUncheckedCreateWithoutDatasetsInput = {
    id?: number
    email: string
    name: string
    degree: string
    field: string
    university: string
    graduationYear: number
    password: string
    createdAt?: Date | string
    updatedAt?: Date | string
    runs?: TrainingRunUncheckedCreateNestedManyWithoutUserInput
  }

  export type UserCreateOrConnectWithoutDatasetsInput = {
    where: UserWhereUniqueInput
    create: XOR<UserCreateWithoutDatasetsInput, UserUncheckedCreateWithoutDatasetsInput>
  }

  export type UserUpsertWithoutDatasetsInput = {
    update: XOR<UserUpdateWithoutDatasetsInput, UserUncheckedUpdateWithoutDatasetsInput>
    create: XOR<UserCreateWithoutDatasetsInput, UserUncheckedCreateWithoutDatasetsInput>
    where?: UserWhereInput
  }

  export type UserUpdateToOneWithWhereWithoutDatasetsInput = {
    where?: UserWhereInput
    data: XOR<UserUpdateWithoutDatasetsInput, UserUncheckedUpdateWithoutDatasetsInput>
  }

  export type UserUpdateWithoutDatasetsInput = {
    email?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    degree?: StringFieldUpdateOperationsInput | string
    field?: StringFieldUpdateOperationsInput | string
    university?: StringFieldUpdateOperationsInput | string
    graduationYear?: IntFieldUpdateOperationsInput | number
    password?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    runs?: TrainingRunUpdateManyWithoutUserNestedInput
  }

  export type UserUncheckedUpdateWithoutDatasetsInput = {
    id?: IntFieldUpdateOperationsInput | number
    email?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    degree?: StringFieldUpdateOperationsInput | string
    field?: StringFieldUpdateOperationsInput | string
    university?: StringFieldUpdateOperationsInput | string
    graduationYear?: IntFieldUpdateOperationsInput | number
    password?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    runs?: TrainingRunUncheckedUpdateManyWithoutUserNestedInput
  }

  export type TrainingRunCreateManyUserInput = {
    id: string
    name?: string
    description?: string
    status: string
    datasetUri: string
    modelPath?: string | null
    resultsPath?: string | null
    metrics?: NullableJsonNullValueInput | InputJsonValue
    parameters?: NullableJsonNullValueInput | InputJsonValue
    geneDepmapResults?: NullableJsonNullValueInput | InputJsonValue
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type DatasetCreateManyUserInput = {
    id: string
    filePath: string
    name: string
    description: string
    rowCount?: number | null
    columnCount?: number | null
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type TrainingRunUpdateWithoutUserInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: StringFieldUpdateOperationsInput | string
    status?: StringFieldUpdateOperationsInput | string
    datasetUri?: StringFieldUpdateOperationsInput | string
    modelPath?: NullableStringFieldUpdateOperationsInput | string | null
    resultsPath?: NullableStringFieldUpdateOperationsInput | string | null
    metrics?: NullableJsonNullValueInput | InputJsonValue
    parameters?: NullableJsonNullValueInput | InputJsonValue
    geneDepmapResults?: NullableJsonNullValueInput | InputJsonValue
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type TrainingRunUncheckedUpdateWithoutUserInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: StringFieldUpdateOperationsInput | string
    status?: StringFieldUpdateOperationsInput | string
    datasetUri?: StringFieldUpdateOperationsInput | string
    modelPath?: NullableStringFieldUpdateOperationsInput | string | null
    resultsPath?: NullableStringFieldUpdateOperationsInput | string | null
    metrics?: NullableJsonNullValueInput | InputJsonValue
    parameters?: NullableJsonNullValueInput | InputJsonValue
    geneDepmapResults?: NullableJsonNullValueInput | InputJsonValue
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type TrainingRunUncheckedUpdateManyWithoutUserInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: StringFieldUpdateOperationsInput | string
    status?: StringFieldUpdateOperationsInput | string
    datasetUri?: StringFieldUpdateOperationsInput | string
    modelPath?: NullableStringFieldUpdateOperationsInput | string | null
    resultsPath?: NullableStringFieldUpdateOperationsInput | string | null
    metrics?: NullableJsonNullValueInput | InputJsonValue
    parameters?: NullableJsonNullValueInput | InputJsonValue
    geneDepmapResults?: NullableJsonNullValueInput | InputJsonValue
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type DatasetUpdateWithoutUserInput = {
    id?: StringFieldUpdateOperationsInput | string
    filePath?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: StringFieldUpdateOperationsInput | string
    rowCount?: NullableIntFieldUpdateOperationsInput | number | null
    columnCount?: NullableIntFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type DatasetUncheckedUpdateWithoutUserInput = {
    id?: StringFieldUpdateOperationsInput | string
    filePath?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: StringFieldUpdateOperationsInput | string
    rowCount?: NullableIntFieldUpdateOperationsInput | number | null
    columnCount?: NullableIntFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type DatasetUncheckedUpdateManyWithoutUserInput = {
    id?: StringFieldUpdateOperationsInput | string
    filePath?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: StringFieldUpdateOperationsInput | string
    rowCount?: NullableIntFieldUpdateOperationsInput | number | null
    columnCount?: NullableIntFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }



  /**
   * Batch Payload for updateMany & deleteMany & createMany
   */

  export type BatchPayload = {
    count: number
  }

  /**
   * DMMF
   */
  export const dmmf: runtime.BaseDMMF
}